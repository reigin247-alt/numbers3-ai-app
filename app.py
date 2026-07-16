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

st.set_page_config(page_title="ナンバーズ3 AI予測アプリ (ルーレット版)", page_icon="🔮", layout="centered")


# --- ルーレット定義（ユーザー指定） ---
HUNDRED_ROULETTE = ['0','1','2','3','4','5','6','7','8','9']
TEN_ROULETTE     = ['0','7','4','1','8','5','2','9','6','3']
ONE_ROULETTE     = ['0','9','8','7','6','5','4','3','2','1']


# --- ユーティリティ（ルーレット基準のズレ・変換） ---
def _shortest_deviation_on_roulette(prev_digit_char: str, curr_digit_char: str, roulette: list) -> str:
    """ルーレット上での最短移動を '右n' / '左n' / '0' で返す"""
    prev = str(prev_digit_char)
    curr = str(curr_digit_char)
    try:
        i_prev = roulette.index(prev)
        i_curr = roulette.index(curr)
    except ValueError:
        return "0"
    n = len(roulette)
    right_dist = (i_curr - i_prev) % n
    left_dist = (i_prev - i_curr) % n
    if right_dist == 0:
        return "0"
    # 最短方向（同数なら右を優先）
    if right_dist <= left_dist:
        return f"右{right_dist}"
    else:
        return f"左{left_dist}"


def shortest_deviation_digit(prev_dig, curr_dig, position: str = "hundred") -> str:
    """
    互換性ラッパー：
    position: "hundred" / "ten" / "one"
    prev_dig / curr_dig: 数字文字列か数値（例: '3' または 3）
    """
    if position == "hundred":
        roulette = HUNDRED_ROULETTE
    elif position == "ten":
        roulette = TEN_ROULETTE
    else:
        roulette = ONE_ROULETTE
    return _shortest_deviation_on_roulette(prev_dig, curr_dig, roulette)


def convert_deviation_to_number(base_digit, deviation_text: str, position: str = "hundred") -> str:
    """
    ルーレット上で base_digit から deviation_text の移動を適用して新しい桁（文字）を返す。
    戻り値は文字列（例: '3'）。
    position: "hundred"/"ten"/"one"
    """
    if position == "hundred":
        roulette = HUNDRED_ROULETTE
    elif position == "ten":
        roulette = TEN_ROULETTE
    else:
        roulette = ONE_ROULETTE

    base = str(base_digit)
    try:
        i_base = roulette.index(base)
    except ValueError:
        return base

    if deviation_text == "0":
        return base

    direction = deviation_text[0]
    try:
        val = int(deviation_text[1:])
    except Exception:
        return base

    n = len(roulette)
    if direction == "右":
        new_idx = (i_base + val) % n
    elif direction == "左":
        new_idx = (i_base - val) % n
    else:
        return base
    return roulette[new_idx]


# --- データ準備（CSVがなければシミュレーション生成） ---
def ensure_data(csv_path: str, min_rows: int = 80) -> str:
    """
    CSVが存在し十分な行数があれば 'real' を返す。
    それ以外はシミュレーションデータを作成して保存し 'simulated' を返す。
    """
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path, encoding="utf-8")
            if len(df) >= min_rows:
                return "real"
        except Exception:
            # 読み込み失敗は続行してシミュレーション作成
            pass

    # シミュレーションデータ生成（ルーレット順は考慮していないが、生データとして桁は0-9）
    random.seed(int(datetime.now().timestamp()))
    records = []
    n = max(min_rows + 5, 100)
    nums = [f"{random.randint(0,9)}{random.randint(0,9)}{random.randint(0,9)}" for _ in range(n)]
    for i in range(len(nums) - 1):
        prev = nums[i]
        curr = nums[i + 1]
        w = i % 5  # 月〜金の割当
        h = shortest_deviation_digit(prev[0], curr[0], position="hundred")
        t = shortest_deviation_digit(prev[1], curr[1], position="ten")
        o = shortest_deviation_digit(prev[2], curr[2], position="one")
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
    le.fit(all_patterns)

    series = df[target_col].astype(str).tolist()
    encoded = []
    for v in series:
        try:
            encoded.append(int(le.transform([v])[0]))
        except Exception:
            encoded.append(int(le.transform(["0"])[0]))

    weekdays = df["曜日"].astype(int).tolist()

    X, y = [], []
    for i in range(len(encoded) - look_back):
        feat = encoded[i: i + look_back] + [weekdays[i + look_back]]
        X.append(feat)
        y.append(encoded[i + look_back])

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
        # 学習データが不足する場合の簡易対処
        if len(X) < 10:
            # 最低限の形状を満たすダミーサンプルを作る
            if X.size == 0:
                X = np.tile(latest_features.reshape(1, -1), (20, 1))
            else:
                idx = np.random.randint(0, max(1, len(X)), size=20)
                X = np.tile(latest_features.reshape(1, -1), (len(idx), 1))
            y = np.random.choice(list(range(len(le.classes_))), size=len(X))

        model = LGBMClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)

        proba = model.predict_proba(latest_features.reshape(1, -1))[0]
        top3_indices = np.argsort(proba)[::-1][:3]

        for idx in top3_indices:
            label_value = model.classes_[idx]
            pattern_text = le.inverse_transform([int(label_value)])[0]
            probability = float(proba[idx]) * 100.0
            base_digit = last_actual_number[i]
            pos = "hundred" if i == 0 else "ten" if i == 1 else "one"
            target_digit = convert_deviation_to_number(base_digit, pattern_text, position=pos)
            digit_candidates[i].append({"digit": str(target_digit), "dev": pattern_text, "proba": probability})

    # 結果整形（本命・対抗・大穴）
    predictions = {}
    types = ["🎯 本命 (第1候補)", "⚔️ 対抗 (第2候補)", "💎 大穴 (第3候補)"]
    for rank in range(3):
        # 安全のため存在チェック
        try:
            d0 = digit_candidates[0][rank]
            d1 = digit_candidates[1][rank]
            d2 = digit_candidates[2][rank]
        except IndexError:
            # 足りない場合は '0' で埋める
            d0 = digit_candidates[0][rank] if rank < len(digit_candidates[0]) else {"digit":"0","dev":"0","proba":0.0}
            d1 = digit_candidates[1][rank] if rank < len(digit_candidates[1]) else {"digit":"0","dev":"0","proba":0.0}
            d2 = digit_candidates[2][rank] if rank < len(digit_candidates[2]) else {"digit":"0","dev":"0","proba":0.0}

        num_str = d0["digit"] + d1["digit"] + d2["digit"]
        avg_proba = (d0["proba"] + d1["proba"] + d2["proba"]) / 3.0
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
st.title("🔮 ナンバーズ3 AI予測システム（ルーレット版）")
st.markdown("指定のルーレット順に基づいて桁ごとのズレを計算・変換し、LightGBMで上位3候補を出します。CSVがなければ内部でシミュレーションデータを生成します。")

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
