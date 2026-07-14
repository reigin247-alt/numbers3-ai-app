import streamlit as st
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
import os
import time
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup

CSV_FILE = "numbers3_directional_deviation.csv"
HISTORY_FILE = "predictions_history.txt"

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
    direction = deviation_text
    val = int(deviation_text[1:])
    if direction == "右": return (base + val) % 10
    elif direction == "左": return (base - val) % 10
    return base

def get_two_lottery_info():
    """『次回』と『次々回』の抽選日と曜日を算出する"""
    now = datetime.now()
    
    target1 = now
    if now.hour >= 19:
        target1 += timedelta(days=1)
    while target1.weekday() >= 5:
        target1 += timedelta(days=1)
        
    target2 = target1 + timedelta(days=1)
    while target2.weekday() >= 5:
        target2 += timedelta(days=1)
        
    weekday_labels = ["月", "火", "水", "木", "金"]
    
    info1 = {"date": target1.strftime("%m月%d日"), "w_str": weekday_labels[target1.weekday()], "w_idx": target1.weekday()}
    info2 = {"date": target2.strftime("%m月%d日"), "w_str": weekday_labels[target2.weekday()], "w_idx": target2.weekday()}
    return info1, info2

# --- ① データ取得（完全防衛モード） ---
def scrape_mizuho_data():
    current_date = datetime.now()
    year, month = current_date.year, current_date.month
    records = []
    
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    try:
        for step in range(12):
            status_text.text(f"🌐 みずほ銀行公式HP: {year}年{month}月のデータを解析中...")
            progress_bar.progress(int((step + 1) / 12 * 100))
            
            url = f"https://mizuhobank.co.jp{year}&month={month}"
            response = requests.get(url, headers=headers, timeout=5)
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
            month -= 1
            if month == 0: month = 12; year -= 1
            time.sleep(0.3)
    except:
        pass
        
    status_text.empty()
    progress_bar.empty()
    
    if len(records) >= 15:
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
        return "real"
    else:
        last_base_num = "549"
        if os.path.exists(CSV_FILE):
            try:
                old_df = pd.read_csv(CSV_FILE, encoding="utf-8")
                if len(old_df) > 0: last_base_num = str(old_df.iloc[-1]["現当選番号"]).zfill(3)
            except: pass
                
        np.random.seed(int(time.time()))
        simulated_nums = [f"{np.random.randint(0,10)}{np.random.randint(0,10)}{np.random.randint(0,10)}" for _ in range(101)]
        simulated_nums[-1] = last_base_num
        
        data_list = []
        for i in range(100):
            prev, curr = simulated_nums[i], simulated_nums[i+1]
            data_list.append({
                "前当選番号": prev, "現当選番号": curr, "曜日": i % 5,
                "百の位_ずれ": calculate_shortest_deviation(prev, curr),
                "十の位_ずれ": calculate_shortest_deviation(prev, curr),
                "一の位_ずれ": calculate_shortest_deviation(prev, curr)
            })
        pd.DataFrame(data_list).to_csv(CSV_FILE, index=False, encoding="utf-8")
        return "simulated"

# --- ② AI予測コア機能 ---
def prepare_ai_data(df, target_col, next_weekday_idx):
    le = LabelEncoder()
    all_patterns = ["左4", "左3", "左2", "左1", "0", "右1", "右2", "右3", "右4", "右5"]
    le.fit(all_patterns)
    
    # 完全に1つの値（スカラー）にするためにリスト内包表記で確実に変換
    raw_series = df[target_col].values
    encoded_list = []
    for val in raw_series:
        if val in le.classes_:
            encoded_list.append(int(le.transform([val])[0]))
        else:
            encoded_list.append(int(le.transform(["0"])[0]))
            
    weekdays = df["曜日"].values
    
    look_back = 3
    X, y = [], []
    for i in range(len(encoded_list) - look_back):
        features = [int(encoded_list[i]), int(encoded_list[i+1]), int(encoded_list[i+2]), int(weekdays[i+3])]
        X.append(features)
        y.append(int(encoded_list[i+3]))
        
    latest_features = [int(encoded_list[-3]), int(encoded_list[-2]), int(encoded_list[-1]), int(next_weekday_idx)]
    
    # 【対策箇所】NumPyの配列作成時に型と形状を完全に揃えて生成
    return np.array(X, dtype=np.int32), np.array(y, dtype=np.int32), le, np.array(latest_features, dtype=np.int32)

def predict_single_step(df, base_number, weekday_idx):
    columns = ["百の位_ずれ", "十の位_ずれ", "一の位_ずれ"]
    digit_candidates = [[], [], []]
    
    for i, col in enumerate(columns):
        X, y, le, latest_features = prepare_ai_data(df, col, weekday_idx)
        model = LGBMClassifier(n_estimators=50, random_state=42, verbose=-1)
        model.fit(X, y)
        
        pred_proba = model.predict_proba(latest_features.reshape(1, -1))[0]
        # クラスラベルに応じた正確な上位3つを取得
        top3_classes_indices = np.argsort(pred_proba)[::-1][:3]
        
        base_digit = base_number[i]
        for idx_in_proba in top3_classes_indices:
            # 予測されたクラスのインデックスから、実際の文字列（右2など）にデコード
            actual_class_index = model.classes_[idx_in_proba]
            pattern_text = le.inverse_transform(np.array([actual_class_index]))[0]
            probability = pred_proba[idx_in_proba] * 100
            
            target_digit = convert_deviation_to_number(base_digit, pattern_text)
            digit_candidates[i].append({"digit": str(target_digit), "dev": pattern_text, "proba": probability})

    predictions = {}
    types = ["🎯 本命", "⚔️ 对抗", "💎 大穴"]
    for rank in range(3):
        num_str = digit_candidates[rank]["digit"] + digit_candidates[rank]["digit"] + digit_candidates[rank]["digit"]
        avg_proba = (digit_candidates[rank]["proba"] + digit_candidates[rank]["proba"] + digit_candidates[rank]["proba"]) / 3
        dev_info = f"百:{digit_candidates[rank]['dev']} 十:{digit_candidates[rank]['dev']} 一:{digit_candidates[rank]['dev']}"
        predictions[types[rank]] = (num_str, avg_proba, dev_info)
    return predictions

# --- ③ UI画面の構成 ---
st.title("🔮 ナンバーズ3 AIダブル予測システム")
st.markdown("曜日・時系列補正をかけたLightGBMモデルを用いて、**次回**および**次々回（次の日）**の2日分の購入候補を先回りして一挙予測します。")

info1, info2 = get_two_lottery_info()

if st.button("🚀 最新データを同期して2日分の予測を開始", type="primary", use_container_width=True):
    with st.spinner("データの同期・AI連続解析を実行中..."):
        mode = scrape_mizuho_data()
        df_main = pd.read_csv(CSV_FILE, encoding="utf-8")
        
        # 1. 次回の予測を実行
        last_actual_number = str(df_main.iloc[-1]["現当選番号"]).zfill(3)
        preds_1 = predict_single_step(df_main, last_actual_number, info1["w_idx"])
        
        # 2. 次々回（次の日）の予測を実行
        next_assumed_num = preds_1["🎯 本命"][0]
        dev_h = calculate_shortest_deviation(last_actual_number, next_assumed_num)
        dev_t = calculate_shortest_deviation(last_actual_number, next_assumed_num)
        dev_o = calculate_shortest_deviation(last_actual_number, next_assumed_num)
        
        new_row = pd.DataFrame([{
            "前当選番号": last_actual_number, "現当選番号": next_assumed_num, "曜日": info1["w_idx"],
            "百の位_ずれ": dev_h, "十の位_ずれ": dev_t, "一の位_ずれ": dev_o
        }])
        df_extended = pd.concat([df_main, new_row], ignore_index=True)
        preds_2 = predict_single_step(df_extended, next_assumed_num, info2["w_idx"])

    # 履歴ログへの保存
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_text = f"=== AIダブル予測ログ : {now_str} ===\n"
    log_text += f"①次回 【{info1['date']}({info1['w_str']})】 ベース: {last_actual_number} -> 本命:{preds_1['🎯 本命'][0]} / 対抗:{preds_1['⚔️ 对抗'][0]} / 大穴:{preds_1['💎 大穴'][0]}\n"
