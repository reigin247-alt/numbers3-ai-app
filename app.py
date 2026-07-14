import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup

CSV_FILE = "numbers3_directional_deviation.csv"
HISTORY_FILE = "predictions_history.txt"

st.set_page_config(page_title="ナンバーズ3 AI予測アプリ", page_icon="🔮", layout="centered")

# --- 固定ルール定義 ---
ALL_PATTERNS = ["左4", "左3", "左2", "左1", "0", "右1", "右2", "右3", "右4", "右5"]

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

# --- ① データ取得（完全防衛型） ---
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
            time.sleep(0.1)
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

# --- ② 確率計算エンジン ---
def predict_single_step_pure(df, base_number, next_weekday_idx):
    columns = ["百の位_ずれ", "十の位_ずれ", "一の位_ずれ"]
    hundreds_cand, tens_cand, ones_cand = [], [], []
    
    for i, col in enumerate(columns):
        series = df[col].astype(str).values.tolist()
        weekdays = df["曜日"].values.tolist()
        last3 = series[-3:]
        
        counts = {p: 0 for p in ALL_PATTERNS}
        total_matched = 0
        
        for k in range(len(series) - 3):
            if series[k:k+3] == last3 and int(weekdays[k+3]) == int(next_weekday_idx):
                next_val = series[k+3]
                if next_val in counts:
                    counts[next_val] += 1
                    total_matched += 1
                    
        if total_matched < 2:
            for k in range(len(series)):
                if int(weekdays[k]) == int(next_weekday_idx):
                    val = series[k]
                    if val in counts:
                        counts[val] += 1
                        total_matched += 1
                        
        probabilities = {}
        for p in ALL_PATTERNS:
            probabilities[p] = (counts[p] / total_matched if total_matched > 0 else 1.0 / 10.0)
            
        sorted_patterns = sorted(probabilities.items(), key=lambda x: x, reverse=True)[:3]
        
        base_digit = base_number[i]
        for pattern_text, proba_val in sorted_patterns:
            target_digit = convert_deviation_to_number(base_digit, pattern_text)
            item = {"digit": str(target_digit), "dev": pattern_text, "proba": float(proba_val * 100)}
            if i == 0: hundreds_cand.append(item)
            elif i == 1: tens_cand.append(item)
            elif i == 2: ones_cand.append(item)

    # 【インデントを4マス単位に完全に揃えてバグを根絶】
    h0, t0, o0 = hundreds_cand[0], tens_cand[0], ones_cand[0]
    h1, t1, o1 = hundreds_cand[1], tens_cand[1], ones_cand[1]
    h2, t2, o2 = hundreds_cand[2], tens_cand[2], ones_cand[2]
    
    res_honmei = {"num": h0["digit"]+t0["digit"]+o0["digit"], "proba": (h0["proba"]+t0["proba"]+o0["proba"])/3, "dev": f"百:{h0['dev']} 十:{t0['dev']} 一:{o0['dev']}"}
    res_taikou = {"num": h1["digit"]+t1["digit"]+o1["digit"], "proba": (h1["proba"]+t1["proba"]+o1["proba"])/3, "dev": f"百:{h1['dev']} 十:{t1['dev']} 一:{o1['dev']}"}
    res_oana   = {"num": h2["digit"]+t2["digit"]+o2["digit"], "proba": (h2["proba"]+t2["proba"]+o2["proba"])/3, "dev": f"百:{h2['dev']} 十:{t2['dev']} 一:{o2['dev']}"}
    
    return {"本命": res_honmei, "対抗": res_taikou, "大穴": res_oana}

# --- 安全な独立ログ書き込み命令 ---
def save_prediction_history_safely(date1, num1, date2, num2, last_num):
    try:
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_text = f"=== AIダブル予測ログ : {now_str} ===\n①次回 【{date1}】 ベース: {last_num} -> 本命:{num1}\n②次々回【{date2}】 ベース: {num1} -> 本命:{num2}\n\n"
        open(HISTORY_FILE, "a", encoding="utf-8").write(log_text)
    except:
        pass

# --- ③ UI画面の構成 ---
st.title("🔮 ナンバーズ3 AIダブル予測システム")
st.markdown("曜日補正・時系列展開モデルを用いて、**次回**および**次々回（次の日）**の2日分の購入候補を一挙予測します。")

info1, info2 = get_two_lottery_info()

if "calculated" not in st.session_state:
    st.session_state.calculated = False
    st.session_state.mode = ""
    st.session_state.last_num = ""
    st.session_state.preds1 = None
    st.session_state.preds2 = None
    st.session_state.next_num = ""

if st.button("🚀 最新データを同期して2日分の予測を開始", type="primary", use_container_width=True):
    with st.spinner("統計確率モデルをロード・連続解析中..."):
        st.session_state.mode = scrape_mizuho_data()
        df_main = pd.read_csv(CSV_FILE, encoding="utf-8")
        
        # 1. 次回の予測を実行
        st.session_state.last_num = str(df_main.iloc[-1]["現当選番号"]).zfill(3)
        st.session_state.preds1 = predict_single_step_pure(df_main, st.session_state.last_num, info1["w_idx"])
        
        # 辞書から安全に本命数字を抽出
        st.session_state.next_num = st.session_state.preds1["本命"]["num"]
        
        # 2. 次々回（次の日）の予測を実行
        st.session_state.preds2 = predict_single_step_pure(df_main, st.session_state.next_num, info2["w_idx"])
        
        # 履歴を安全に保存
        save_prediction_history_safely(
            info1["date"], st.session_state.next_num, 
            info2["date"], st.session_state.preds2["本命"]["num"], 
            st.session_state.last_num
        )
            
        st.session_state.calculated = True

# --- 画面表示エリア ---
if st.session_state.calculated:
    if st.session_state.mode == "real":
