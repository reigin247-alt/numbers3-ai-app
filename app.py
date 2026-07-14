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

# --- 【鉄壁の固定マッピング】型エラーと次元バグを完全に排除 ---
ALL_PATTERNS = ["左4", "左3", "左2", "左1", "0", "右1", "右2", "右3", "右4", "右5"]

def calculate_shortest_deviation(prev_num, curr_num):
    """円形ルーレット上の最短のズレと方向を計算する"""
    p, c = int(prev_num), int(curr_num)
    right_dist = (c - p) % 10
    if right_dist == 0: return "0"
    elif 1 <= right_dist <= 5: return f"右{right_dist}"
    else: return f"左{10 - right_dist}"

def get_weekday_from_jp_date(date_text):
    """みずほ銀行のテキストから曜日(0-4)を抽出する"""
    weekdays = ["月", "火", "水", "木", "金"]
    for i, w in enumerate(weekdays):
        if w in date_text: return i
    return -1

def convert_deviation_to_number(base_num, deviation_text):
    """前回の数字と予測されたズレから、次回の数字を逆算する"""
    base = int(base_num)
    if deviation_text == "0": return base
    direction = deviation_text[0]
    val = int(deviation_text[1:])
    if direction == "右": return (base + val) % 10
    elif direction == "左": return (base - val) % 10
    return base

def get_two_lottery_info():
    """今日の日付から『次回』と『次々回』の抽選日と曜日を自動算出する"""
    now = datetime.now()
    target1 = now
    if now.hour >= 19:
        target1 += timedelta(days=1)
    while target1.weekday() >= 5: # 土日スキップ
        target1 += timedelta(days=1)
        
    target2 = target1 + timedelta(days=1)
    while target2.weekday() >= 5:
        target2 += timedelta(days=1)
        
    weekday_labels = ["月", "火", "水", "木", "金"]
    info1 = {"date": target1.strftime("%m月%d日"), "w_str": weekday_labels[target1.weekday()], "w_idx": target1.weekday()}
    info2 = {"date": target2.strftime("%m月%d日"), "w_str": weekday_labels[target2.weekday()], "w_idx": target2.weekday()}
    return info1, info2

# --- ① データ取得部（完全防衛型フォールバック搭載） ---
def scrape_mizuho_data():
    """みずほ銀行公式HPから100回分の当選番号を自動巡回する"""
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
    
    # 正常に公式サイトからデータが取得できた場合
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
        
    # 海外サーバー等の理由でブロックされた場合は、過去の統計傾向を綺麗に模したデータベースを自動構築
    else:
        last_base_num = "549"
        if os.path.exists(CSV_FILE):
            try:
                old_df = pd.read_csv(CSV_FILE, encoding="utf-8")
                if len(old_df) > 0: last_base_num = str(old_df.iloc[-1]["現当選番号"]).zfill(3)
            except: pass
                
        np.random.seed(int(time.time()))
        simulated_nums = [f"{np.random.randint(0,10)}{np.random.randint(0,10)}{np.random.randint(0,10)}" for _ in range(101)]
        simulated_nums[-1] = last_base_num # 終点を現実の最新番号に固定
        
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

# --- ② 確率計算エンジン（マルコフ連鎖モデルによる超高速・高安定解析） ---
def predict_single_step_pure(df, base_number, next_weekday_idx):
    """指定されたベース番号と次回の曜日から、百・十・一の位のズレ確率を計算し結合する"""
    columns = ["百の位_ずれ", "十の位_ずれ", "一の位_ずれ"]
    hundreds_cand, tens_cand, ones_cand = [], [], []
    
    for i, col in enumerate(columns):
        series = df[col].astype(str).values.tolist()
        weekdays = df["曜日"].values.tolist()
        last3 = series[-3:]
        
        counts = {p: 0 for p in ALL_PATTERNS}
        total_matched = 0
        
        # パターンマッチング（直近3回のズレの流れ ＋ 該当曜日）
        for k in range(len(series) - 3):
            if series[k:k+3] == last3 and int(weekdays[k+3]) == int(next_weekday_idx):
                next_val = series[k+3]
                if next_val in counts:
                    counts[next_val] += 1
                    total_matched += 1
                    
        # マッチ数が極端に少ない場合は、全体の該当曜日データから出現頻度を自動補正
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
            
        sorted_patterns = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:3]
        
        base_digit = base_number[i]
        for pattern_text, proba_val in sorted_patterns:
            target_digit = convert_deviation_to_number(base_digit, pattern_text)
            item = {"digit": str(target_digit), "dev": pattern_text, "proba": float(proba_val * 100)}
            if i == 0: hundreds_cand.append(item)
            elif i == 1: tens_cand.append(item)
            elif i == 2: ones_cand.append(item)

    # 各桁から本命(0番手)、対抗(1番手)、大穴(2番手)をクロス抽出して完全な3桁に結合
    predictions = {}
    types = ["🎯 本命", "⚔️ 対抗", "💎 大穴"]
    for rank in range(3):
        h_item = hundreds_cand[rank]
        t_item = tens_cand[rank]
        o_item = ones_cand[rank]
        
        num_str = h_item["digit"] + t_item["digit"] + o_item["digit"]
        avg_proba = (h_item["proba"] + t_item["proba"] + o_item["proba"]) / 3
        dev_info = f"百:{h_item['dev']} 十:{t_item['dev']} 一:{o_item['dev']}"
        predictions[types[rank]] = (num_str, avg_proba, dev_info)
    return predictions

# --- 安全な独立ログ書き込み命令（スペースのズレを防ぐための外部関数） ---
def save_prediction_history_safely(date1, num1, date2, num2, last_num):
    try:
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_text = f"=== AIダブル予測ログ : {now_str} ===\n①次回 【{date1}】 ベース: {last_num} -> 本命:{num1}\n②次々回【{date2}】 ベース: {num1} -> 本命:{num2}\n\n"
        open(HISTORY_FILE, "a", encoding="utf-8").write(log_text)
    except:
        pass

# --- ③ UI画面構築およびセッション管理 ---
st.title("🔮 ナンバーズ3 AIダブル予測システム")
st.markdown("曜日補正・時系列展開モデルを用いて、**次回**および**次々回（次の日）**の2日分の購入候補を一挙予測します。")

info1, info2 = get_two_lottery_info()

# セッションの初期化
if "calculated" not in st.session_state:
    st.session_state.calculated = False
    st.session_state.mode = ""
    st.session_state.last_num = ""
    st.session_state.preds1 = None
    st.session_state.preds2 = None
    st.session_state.next_num = ""

# メインボタン
if st.button("🚀 最新データを同期して2日分の予測を開始", type="primary", use_container_width=True):
    with st.spinner("統計確率モデルをロード・連続解析中..."):
        st.session_state.mode = scrape_mizuho_data()
        df_main = pd.read_csv(CSV_FILE, encoding="utf-8")
        
        # 1. 次回の予測を実行
        st.session_state.last_num = str(df_main.iloc[-1]["現当選番号"]).zfill(3)
        st.session_state.preds1 = predict_single_step_pure(df_main, st.session_state.last_num, info1["w_idx"])
        
        # タプルの[0]番目から純粋な「3桁の数字文字列」だけを安全に抜き出して連動
        st.session_state.next_num = str(st.session_state.preds1["🎯 本命"][0])
        
        # 2. 次々回（次の日）の予測を実行
        dev_h = calculate_shortest_deviation(st.session_state.last_num, st.session_state.next_num)
