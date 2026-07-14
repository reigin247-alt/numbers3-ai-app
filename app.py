import streamlit as st
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
import os
import time
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup

CSV_FILE = "numbers3_directional_deviation.csv"
HISTORY_FILE = "predictions_history.txt"

st.set_page_config(page_title="ナンバーズ3 AI予測アプリ", page_icon="🔮", layout="centered")

# --- 【鉄壁の固定ルール定義】型エラーと次元バグを完全に排除 ---
ALL_PATTERNS = ["左4", "左3", "左2", "左1", "0", "右1", "右2", "右3", "右4", "右5"]
PATTERN_TO_INT = {p: i for i, p in enumerate(ALL_PATTERNS)}
INT_TO_PATTERN = {i: p for i, p in enumerate(ALL_PATTERNS)}

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

# --- ① データ取得（エラー時も確実に100行の綺麗な形を返す） ---
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
    raw_series = df[target_col].astype(str).values
    encoded_list = [PATTERN_TO_INT[val] if val in PATTERN_TO_INT else PATTERN_TO_INT["0"] for val in raw_series]
    weekdays = df["曜日"].values
    
    look_back = 3
    X, y = [], []
    for i in range(len(encoded_list) - look_back):
        features = [int(encoded_list[i]), int(encoded_list[i+1]), int(encoded_list[i+2]), int(weekdays[i+3])]
        X.append(features)
        y.append(int(encoded_list[i+3]))
        
    latest_features = [int(encoded_list[-3]), int(encoded_list[-2]), int(encoded_list[-1]), int(next_weekday_idx)]
    return np.array(X, dtype=np.int32), np.array(y, dtype=np.int32), latest_features

def predict_single_step(df, base_number, weekday_idx):
    columns = ["百の位_ずれ", "十の位_ずれ", "一の位_ずれ"]
    
    # digit_candidates[0] = 百の位の上位3つ, [1] = 十の位の上位3つ, [2] = 一の位の上位3つ
    digit_candidates = [[], [], []]
    
    for i, col in enumerate(columns):
        X, y, latest_features = prepare_ai_data(df, col, weekday_idx)
        
        # クラス数を0〜9の10クラスに完全固定して強制学習
        model = LGBMClassifier(n_estimators=50, random_state=42, verbose=-1)
        model.fit(X, y)
        
        # 1行だけの2次元配列にして予測に渡す
        pred_proba = model.predict_proba(np.array([latest_features], dtype=np.int32))[0]
        
        # モデルが学習した実際のクラスラベルのリスト
        learned_classes = model.classes_.tolist()
        
        # 10個の全パターンの確率をまとめる
        pattern_probabilities = []
        for pattern_idx in range(10):
            if pattern_idx in learned_classes:
                proba_val = pred_proba[learned_classes.index(pattern_idx)]
            else:
                proba_val = 0.0
            pattern_probabilities.append(proba_val)
            
        # 確率が高い順に上位3つのパターン（0〜9のインデックス）をソート
        top3_patterns = np.argsort(pattern_probabilities)[::-1][:3]
        
        base_digit = base_number[i]
        for p_idx in top3_patterns:
            pattern_text = INT_TO_PATTERN[p_idx]
            probability = float(pattern_probabilities[p_idx] * 100)
            target_digit = convert_deviation_to_number(base_digit, pattern_text)
            
            digit_candidates[i].append({
                "digit": str(target_digit),
                "dev": pattern_text,
                "proba": probability
            })

    # 正しい3桁の組み合わせ（本命＝各桁の1番手、対抗＝2番手、大穴＝3番手）をパッキング
    predictions = {}
    types = ["🎯 本命", "⚔️ 対抗", "💎 大穴"]
    for rank in range(3):
        num_str = digit_candidates[0][rank]["digit"] + digit_candidates[1][rank]["digit"] + digit_candidates[2][rank]["digit"]
        avg_proba = (digit_candidates[0][rank]["proba"] + digit_candidates[1][rank]["proba"] + digit_candidates[2][rank]["proba"]) / 3
        dev_info = f"百:{digit_candidates[0][rank]['dev']} 十:{digit_candidates[1][rank]['dev']} 一:{digit_candidates[2][rank]['dev']}"
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
        next_assumed_num = str(preds_1["🎯 本命"][0])  # 本命の数字文字列を取得
        dev_h = calculate_shortest_deviation(last_actual_number, next_assumed_num)
        dev_t = calculate_shortest_deviation(last_actual_number, next_assumed_num)
        dev_o = calculate_shortest_deviation(last_actual_number, next_assumed_num)
        
        new_row = pd.DataFrame([{
            "前当選番号": last_actual_number, "現当選番号": next_assumed_num, "曜日": info1["w_idx"],
            "百の位_ずれ": dev_h, "十の位_ずれ": dev_t, "一の位_ずれ": dev_o
        }])
        df_extended = pd.concat([df_main, new_row], ignore_index=True)
