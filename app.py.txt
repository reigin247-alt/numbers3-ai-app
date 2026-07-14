import streamlit as st
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
import os
import time
from datetime import datetime
import requests
from bs4 import BeautifulSoup

CSV_FILE = "numbers3_directional_deviation.csv"
HISTORY_FILE = "predictions_history.txt"

# --- ページ設定 ---
st.set_page_config(page_title="ナンバーズ3 AI予測アプリ", page_icon="🔮", layout="centered")

def calculate_shortest_deviation(prev_num, curr_num):
    p, c = int(prev_num), int(curr_num)
    right_dist = (c - p) % 10
    if right_dist == 0: return "0"
    elif 1 <= right_dist <= 5: return f"右{right_dist}"
    else: return f"左{10 - right_dist}"

def get_weekday_from_jp_date(date_text):
    weekdays = ["月", "火", "水", "木", "金"]
    for i, w in enumerate(weekdays):
        if w in date_text: return i
    return -1

def convert_deviation_to_number(base_num, deviation_text):
    base = int(base_num)
    if deviation_text == "0": return base
    direction = deviation_text[0]
    val = int(deviation_text[1:])
    if direction == "右": return (base + val) % 10
    elif direction == "左": return (base - val) % 10
    return base

# --- ① データスクレイピング機能 ---
def scrape_mizuho_data():
    current_date = datetime.now()
    year, month = current_date.year, current_date.month
    records = []
    
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    for step in range(12):
        status_text.text(f"🌐 みずほ銀行公式HP: {year}年{month}月のデータを解析中...")
        progress_bar.progress(int((step + 1) / 12 * 100))
        
        url = f"https://mizuhobank.co.jp{year}&month={month}"
        try:
            response = requests.get(url)
            response.encoding = 'shift_jis'
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                tables = soup.find_all('table', class_='typeTK')
                for table in tables:
                    rows = table.find_all('tr')
                    date_text, num_text = "", ""
                    for row in rows:
                        th = row.find('th')
                        td = row.find('td')
                        if th and '抽選日' in th.text and td: date_text = td.text.strip()
                        if th and ('数字' in th.text or '抽せん数字' in th.text) and td:
                            num_text = td.text.strip().replace(" ", "")[:3]
                    if num_text.isdigit() and len(num_text) == 3:
                        w_idx = get_weekday_from_jp_date(date_text)
                        if w_idx != -1: records.append({"number": num_text, "weekday": w_idx})
            if not tables or len(records) >= 101: break
        except:
            break
        month -= 1
        if month == 0: month = 12; year -= 1
        time.sleep(0.5)
        
    status_text.empty()
    progress_bar.empty()
    
    if len(records) < 2: return False
    
    raw_records = records[:101][::-1]
    data_list = []
    for i in range(len(raw_records) - 1):
        prev, curr = raw_records[i]["number"], raw_records[i+1]["number"]
        data_list.append({
            "前当選番号": prev, "現当選番号": curr, "曜日": raw_records[i+1]["weekday"],
            "百の位_ずれ": calculate_shortest_deviation(prev, curr),
            "十の位_ずれ": calculate_shortest_deviation(prev, curr),
            "一の位_ずれ": calculate_shortest_deviation(prev, curr)
        })
    pd.DataFrame(data_list).to_csv(CSV_FILE, index=False, encoding="utf-8")
    return True

# --- ② AI予測・保存機能 ---
def prepare_ai_data(df, target_col):
    le = LabelEncoder()
    le.fit(["左4", "左3", "左2", "左1", "0", "右1", "右2", "右3", "右4", "右5"])
    
    encoded_series = df[target_col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else le.transform(["0"])[0]).values
    weekdays = df["曜日"].values
    
    look_back = 3
    X, y = [], []
    for i in range(len(encoded_series) - look_back):
        features = list(encoded_series[i : i + look_back]) + [weekdays[i + look_back]]
        X.append(features)
        y.append(encoded_series[i + look_back])
        
    next_weekday = (datetime.now().weekday()) % 5
    latest_features = list(encoded_series[-look_back:]) + [next_weekday]
    return np.array(X), np.array(y), le, np.array(latest_features), next_weekday

def run_prediction():
    df = pd.read_csv(CSV_FILE, encoding="utf-8")
    last_actual_number = str(df.iloc[-1]["現当選番号"]).zfill(3)
    columns = ["百の位_ずれ", "十の位_ずれ", "一の位_ずれ"]
    weekday_labels = ["月", "火", "水", "木", "金"]
    
    digit_candidates = [[], [], []]
    next_w = 0
    
    for i, col in enumerate(columns):
        X, y, le, latest_features, next_w = prepare_ai_data(df, col)
        model = LGBMClassifier(n_estimators=50, random_state=42, verbose=-1)
        model.fit(X, y)
        
        pred_proba = model.predict_proba(latest_features.reshape(1, -1))[0]
        top3_indices = np.argsort(pred_proba)[::-1][:3]
        
        base_digit = last_actual_number[i]
        for idx in top3_indices:
            pattern_text = le.inverse_transform([idx])[0]
            probability = pred_proba[idx] * 100
            target_digit = convert_deviation_to_number(base_digit, pattern_text)
            digit_candidates[i].append({"digit": str(target_digit), "dev": pattern_text, "proba": probability})

    predictions = {}
    types = ["🎯 本命 (第1候補)", "⚔️ 対抗 (第2候補)", "💎 大穴 (第3候補)"]
    for rank in range(3):
        num_str = digit_candidates[0][rank]["digit"] + digit_candidates[1][rank]["digit"] + digit_candidates[2][rank]["digit"]
        avg_proba = (digit_candidates[0][rank]["proba"] + digit_candidates[1][rank]["proba"] + digit_candidates[2][rank]["proba"]) / 3
        dev_info = f"百:{digit_candidates[0][rank]['dev']} 十:{digit_candidates[1][rank]['dev']} 一:{digit_candidates[2][rank]['dev']}"
        predictions[types[rank]] = (num_str, avg_proba, dev_info)

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_text = f"=== AI予測日時 : {now_str} ===\n対象曜日: {weekday_labels[next_w]}曜日 / 前回番号: {last_actual_number}\n"
    for title, (num, proba, dev) in predictions.items():
        log_text += f" {title} -> 【 {num} 】 (信頼度: {proba:.1f}% / {dev})\n"
    log_text += "\n"
    with open(HISTORY_FILE, "a", encoding="utf-8") as f: f.write(log_text)
    
    return last_actual_number, weekday_labels[next_w], predictions

# --- ③ UI画面の構成 ---
st.title("🔮 ナンバーズ3 AI予測システム")
st.markdown("みずほ銀行の公式サイトから最新データを自動巡回し、曜日・時系列補正をかけたLightGBMモデルで上位3つの候補を自動計算します。")

if st.button("🚀 最新データを取得してAI予測を開始", type="primary", use_container_width=True):
    with st.spinner("最新データをスクレイピング中...（約5〜10秒かかります）"):
        success = scrape_mizuho_data()
        
    if success:
        with st.spinner("AIモデルの学習と予測を実行中..."):
            last_num, next_day, preds = run_prediction()
            
        st.success("🎉 AIの分析と予測が完了しました！結果をテキストファイルに自動保存しました。")
        
        st.subheader(f"📊 予測シミュレーション結果（次回【{next_day}曜日】分）")
        st.info(f"💡 前回の当選番号: **{last_num}**")
        
        col1, col2, col3 = st.columns(3)
        for i, (title, (num, proba, dev)) in enumerate(preds.items()):
            target_col = col1 if i == 0 else col2 if i == 1 else col3
            with target_col:
                st.metric(label=title, value=num)
                st.caption(f"🤖 期待値: **{proba:.1f}%**")
                st.caption(f"_{dev}_")
    else:
        st.error("みずほ銀行のサイトからデータを取得できませんでした。時間をおいて再度お試しください。")

st.divider()
st.subheader("📜 過去の予測履歴ログ")
if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        history_data = f.read()
    st.text_area(label="predictions_history.txt の中身", value=history_data, height=200)
else:
    st.caption("まだ予測履歴はありません。上のボタンを押すと自動作成されます。")
