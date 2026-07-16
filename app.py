import streamlit as st
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
import os
from datetime import datetime
import random

CSV_FILE = "numbers3_directional_deviation.csv"
HISTORY_FILE = "predictions_history.txt"

st.set_page_config(page_title="ナンバーズ3 AI予測アプリ (簡易版)", page_icon="🔮", layout="centered")


# --- ユーティリティ ---
def shortest_deviation_digit(prev_dig: int, curr_dig: int) -> str:
    """1桁（0-9）同士の最短ズレを '右n' / '左n' / '0' で返す"""
    right_dist = (curr_dig - prev_dig) % 10
    if right_dist == 0:
        return "0"
    if 1 <= right_dist <= 5:
        return f"右{right_dist}"
    return f"左{10 - right_dist}"


def convert_deviation_to_number(base_digit: str, deviation_text: str) -> int:
    """基準の1桁文字とズレ表現から予測桁（0-9）を返す"""
    base = int(base_digit)
    if deviation_text == "0":
        return base
    direction = deviation_text[0]
    val = int(deviation_text[1:])
    if direction == "右":
        return (base + val) % 10
    if direction == "左":
        return (base - val) % 10
    return base


# --- データ準備（CSVがなければシミュレーション生成） ---
def ensure_data(csv_path: str, min_rows: int = 80) -> str:
    """
    CSVが存在すればロードして返す。存在しない／行数不足ならシミュレーションデータを作成して保存する。
    戻り値: 'real' または 'simulated'
    """
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, encoding="utf-8")
        if len(df) >= min_rows:
            return "real"

    # シミュレーションデータ生成
    random.seed(int(datetime.now().timestamp()))
    records = []
    n = max(min_rows + 5, 100)
    nums = [f"{random.randint(0,9)}{random.randint(0,9)}{random.randint(0,9)}" for _ in range(n)]
    for i in range(len(nums) - 1):
        prev = nums[i]
        curr = nums[i + 1]
        w = i % 5  # 月〜金のシンプルな割当
        h = shortest_deviation_digit(int(prev[0]), int(curr[0]))
        t = shortest_deviation_digit(int(prev[1]), int(curr[1]))
        o = shortest_deviation_digit(int(prev[2]), int(curr[2]))
        records.append({
            "前当選番号": prev,
            "現当選番号": curr,
            "曜日": w,
            "百の位_ずれ": h,
            "十の位_ずれ": t,
            "一の位_ずれ": o
        })
    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    return "simulated"


# --- AI用データ作成 ---
def prepare_ai_data(df: pd.DataFrame, target_col: str, look_back: int = 3):
    """
    target_col（例: '百の位_ずれ'）を基に時系列特徴量を作成する。
    戻り値: X, y, label_encoder, latest_features_array, next_weekday_index
    """
    all_patterns = ["左4", "左3", "左2", "左1", "0", "右1", "右2", "右3", "右4", "右5"]
    le = LabelEncoder()
    le.fit(all_patterns)  # 全パターンを先に定義しておく（未出現パターンも扱えるように）

    series = df[target_col].astype(str).tolist()
    encoded = []
    for v in series:
        try:
            encoded.append(int(le.transform([v])[0]))
        except Exception:
            # 想定外は「0」扱いにする
            encoded.append(int(le.transform(["0"])[0]))

    weekdays = df["曜日"].astype(int).tolist()

    X, y = [], []
    for i in range(len(encoded) - look_back):
        feat = encoded[i: i + look_back] + [weekdays[i + look_back]]
        X.append(feat)
        y.append(encoded[i + look_back])

    # 予測用最新特徴（末尾の look_back 個 + 次回曜日）
    next_weekday = (datetime.now().weekday()) % 5
    latest_features = encoded[-look_back:] + [next_weekday]
    return np.array(X), np.array(y), le, np.array(latest_features), next_weekday


# --- 予測実行 ---
def run_prediction(csv_path: str, mode_text: str):
    df = pd.read_csv(csv_path, encoding="utf-8")
    last_actual_number = str(df.iloc[-1]["現当選番号"]).zfill(3)
    columns = ["百の位_ずれ", "十の位_ずれ", "一の位_ずれ"]
    weekday_labels = ["月", "火", "水", "木", "金"]

    digit_candidates = [[], [], []]
    next_w = 0

    for i, col in enumerate(columns):
        X, y, le, latest_features, next_w = prepare_ai_data(df, col)
        # 学習データが不足する場合は警告的にシミュレーションで埋める（最低行数を確保）
        if len(X) < 10:
            # 簡易に同じデータをリサンプリングして増やす
            idx = np.random.randint(0, max(1, len(X)), size=20)
            X = np.tile(latest_features.reshape(1, -1), (len(idx), 1))
            y = np.random.choice(list(range(len(le.classes_))), size=len(idx))

        model = LGBMClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)

        proba = model.predict_proba(latest_features.reshape(1, -1))[0]
        top3_indices = np.argsort(proba)[::-1][:3]  # probability の上位3つ（インデックスは model.classes_ の列順）

        for idx in top3_indices:
            # model.classes_[idx] が実際のラベル値（encoded）を表す
            label_value = model.classes_[idx]
            # LabelEncoder は label_value -> パターン文字列に逆変換
            pattern_text = le.inverse_transform([int(label_value)])[0]
            probability = proba[idx] * 100
            base_digit = last_actual_number[i]
            target_digit = convert_deviation_to_number(base_digit, pattern_text)
            digit_candidates[i].append({"digit": str(target_digit), "dev": pattern_text, "proba": probability})

    # 結果整形（上から順に本命・対抗・大穴）
    predictions = {}
    types = ["🎯 本命 (第1候補)", "⚔️ 対抗 (第2候補)", "💎 大穴 (第3候補)"]
    for rank in range(3):
        d0 = digit_candidates[0][rank]
        d1 = digit_candidates[1][rank]
        d2 = digit_candidates[2][rank]
        num_str = d0["digit"] + d1["digit"] + d2["digit"]
        avg_proba = (d0["proba"] + d1["proba"] + d2["proba"]) / 3
        dev_info = f"百:{d0['dev']} 十:{d1['dev']} 一:{d2['dev']}"
        predictions[types[rank]] = (num_str, avg_proba, dev_info)

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_text = f"=== AI予測日時 : {now_str} ({mode_text}モード) ===\n対象曜日: {weekday_labels[next_w]}曜日 / 前回番号: {last_actual_number}\n"
    for title, (num, proba, dev) in predictions.items():
        log_text += f" {title} -> 【 {num} 】 (信頼度: {proba:.1f}% / {dev})\n"
    log_text += "\n"
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(log_text)

    return last_actual_number, weekday_labels[next_w], predictions


# --- Streamlit UI ---
st.title("🔮 ナンバーズ3 AI予測システム（簡易・安定版）")
st.markdown("CSVが存在すればそれを使い、なければ内部でシミュレーションデータを生成してLightGBMで上位3候補を出します。外部サイトへのスクレイピングは行いません。")

if st.button("🚀 最新データを用意してAI予測を開始", type="primary", use_container_width=True):
    with st.spinner("データ準備中・AI解析を実行中..."):
        mode = ensure_data(CSV_FILE)
        last_num, next_day, preds = run_prediction(CSV_FILE, mode)

    if mode == "real":
        st.success("🎉 CSVデータを読み込み、AI解析を完了しました。")
    else:
        st.warning("⚡ シミュレーションデータで予測を行いました（CSVが見つからなかったため）。")

    st.subheader(f"📊 予測結果（次回【{next_day}曜日】想定）")
    st.info(f"💡 前回（ベース）の番号: **{last_num}**")

    col1, col2, col3 = st.columns(3)
    for i, (title, (num, proba, dev)) in enumerate(preds.items()):
        target_col = col1 if i == 0 else col2 if i == 1 else col3
        with target_col:
            st.metric(label=title, value=num)
            st.caption(f"🤖 期待値: **{proba:.1f}%**")
            st.caption(f"_{dev}_")

st.divider()
st.subheader("📜 過去の予測履歴ログ")
if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        history_data = f.read()
    st.text_area(label="predictions_history.txt の中身", value=history_data, height=200)
else:
    st.caption("まだ予測履歴はありません。上のボタンを押すと自動作成されます。")
