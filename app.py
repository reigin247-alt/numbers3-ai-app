import streamlit as st
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
import os
from datetime import datetime
import random
from io import StringIO

CSV_FILE = "numbers3_directional_deviation.csv"
HISTORY_FILE = "predictions_history.txt"

st.set_page_config(page_title="ナンバーズ3 AI予測アプリ（ルーレット版・アップロード対応）", page_icon="🔮", layout="centered")


# --- ルーレット定義（ユーザー指定） ---
HUNDRED_ROULETTE = ['0','1','2','3','4','5','6','7','8','9']
TEN_ROULETTE     = ['0','7','4','1','8','5','2','9','6','3']
ONE_ROULETTE     = ['0','9','8','7','6','5','4','3','2','1']


# --- ユーティリティ（ルーレット基準のズレ・変換） ---
def _shortest_deviation_on_roulette(prev_digit_char: str, curr_digit_char: str, roulette: list) -> str:
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
    if right_dist <= left_dist:
        return f"右{right_dist}"
    else:
        return f"左{left_dist}"


def shortest_deviation_digit(prev_dig, curr_dig, position: str = "hundred") -> str:
    if position == "hundred":
        roulette = HUNDRED_ROULETTE
    elif position == "ten":
        roulette = TEN_ROULETTE
    else:
        roulette = ONE_ROULETTE
    return _shortest_deviation_on_roulette(prev_dig, curr_dig, roulette)


def convert_deviation_to_number(base_digit, deviation_text: str, position: str = "hundred") -> str:
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


# --- CSV アップロード処理（ブラウザからのエクスポート CSV を内部形式に変換して保存） ---
def process_uploaded_history_csv(uploaded_file) -> str:
    """
    期待フォーマット（ヘッダの日本語は柔軟）: 回号, 抽せん日, 当せん番号
    作業:
      - CSV を読み込み、回号でソート（古い→新しい）
      - 連続ペアから 前当選番号 / 現当選番号 / 曜日 / 各桁のズレ を計算して CSV_FILE を作成
    戻り値: 'real' on success, raises Exception on parse error
    """
    # uploaded_file は Streamlit UploadedFile オブジェクト
    raw = uploaded_file.getvalue()
    text = raw.decode('utf-8') if isinstance(raw, (bytes, bytearray)) else str(raw)
    df_in = pd.read_csv(StringIO(text), dtype=str)

    # 列名から round,date,num を推定
    cols = list(df_in.columns)
    col_round = next((c for c in cols if "回" in c or "回号" in c or "round" in c.lower()), None)
    col_date  = next((c for c in cols if "日" in c or "抽" in c or "date" in c.lower()), None)
    col_num   = next((c for c in cols if "番" in c or "当" in c or "num" in c.lower()), None)

    if not (col_round and col_date and col_num):
        raise ValueError("CSV の列が見つかりません。'回号/開催回'、'抽せん日'、'当せん番号' を含む CSV をアップロードしてください。")

    # 整形
    df_in = df_in[[col_round, col_date, col_num]].rename(columns={col_round: "回号", col_date: "抽せん日", col_num: "当せん番号"})

    # 回号を数値化（可能なら）、古い→新しいにソート
    try:
        df_in["回号"] = df_in["回号"].astype(int)
        df_in = df_in.sort_values("回号").reset_index(drop=True)
    except Exception:
        df_in = df_in.reset_index(drop=True)

    # 当せん番号をゼロパディングして3桁確保（数字のみ抽出）
    df_in["当せん番号"] = df_in["当せん番号"].astype(str).str.extract(r'(\d{1,3})', expand=False).fillna("000").apply(lambda x: x.zfill(3))

    # 抽せん日を日時パースして曜日を作る（失敗時は index%5）
    try:
        df_in["parsed_date"] = pd.to_datetime(df_in["抽せん日"], errors='coerce')
    except Exception:
        df_in["parsed_date"] = pd.NaT
    df_in["weekday_idx"] = df_in["parsed_date"].dt.weekday  # 0=Mon ... 6=Sun

    def map_weekday(row_idx, wd):
        # 月～金を 0-4 にマップ。NaT は index%5 に
        if pd.isna(wd):
            return int(row_idx % 5)
        w = int(wd)
        return w if w < 5 else int(w % 5)

    df_in["weekday_mapped"] = [map_weekday(i, wd) for i, wd in enumerate(df_in["weekday_idx"])]

    # 連続ペア作成（古い->新しい）
    records = []
    for i in range(len(df_in) - 1):
        prev = df_in.loc[i, "当せん番号"]
        curr = df_in.loc[i + 1, "当せん番号"]
        w_idx = int(df_in.loc[i + 1, "weekday_mapped"])
        h = shortest_deviation_digit(prev[0], curr[0], position="hundred")
        t = shortest_deviation_digit(prev[1], curr[1], position="ten")
        o = shortest_deviation_digit(prev[2], curr[2], position="one")
        records.append({
            "前当選番号": prev,
            "現当選番号": curr,
            "曜日": w_idx,
            "百の位_ずれ": h,
            "十の位_ずれ": t,
            "一の位_ずれ": o
        })

    if not records:
        raise ValueError("CSV の行数が足りません（2行以上必要です）。")

    df_out = pd.DataFrame(records)
    df_out.to_csv(CSV_FILE, index=False, encoding="utf-8")
    return "real"


# --- AI用データ作成（次回＝「直近のデータの次の日」ベース） ---
def prepare_ai_data(df: pd.DataFrame, target_col: str, look_back: int = 3):
    """
    target_col（例: '百の位_ずれ'）を基に時系列特徴量を作成する。
    次回の曜日特徴は「直近レコードの '曜日' + 1」（mod 5）を使用します（つまり次回＝翌回）。
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

    # '曜日' カラムが存在することを前提（ensure_data / upload で作成される）
    if "曜日" in df.columns and len(df["曜日"]) > 0:
        try:
            last_weekday = int(df["曜日"].iloc[-1])
            next_weekday = (last_weekday + 1) % 5  # 直近の次回（翌回）
        except Exception:
            next_weekday = (datetime.now().weekday()) % 5
    else:
        next_weekday = (datetime.now().weekday()) % 5

    weekdays = df["曜日"].astype(int).tolist() if "曜日" in df.columns else [0] * len(encoded)

    # look_back に満たない場合は先頭要素でパディングして最新特徴を作れるようにする
    if len(encoded) < look_back:
        pad_val = encoded[0] if encoded else int(le.transform(["0"])[0])
        encoded_padded = [pad_val] * (look_back - len(encoded)) + encoded
    else:
        encoded_padded = encoded

    X, y = [], []
    for i in range(len(encoded) - look_back):
        feat = encoded[i: i + look_back] + [weekdays[i + look_back]]
        X.append(feat)
        y.append(encoded[i + look_back])

    # 予測用最新特徴（末尾の look_back 個 + 次回（直近の次）曜日）
    latest_window = encoded_padded[-look_back:]
    latest_features = latest_window + [next_weekday]
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
        try:
            d0 = digit_candidates[0][rank]
            d1 = digit_candidates[1][rank]
            d2 = digit_candidates[2][rank]
        except IndexError:
            d0 = digit_candidates[0][rank] if rank < len(digit_candidates[0]) else {"digit":"0","dev":"0","proba":0.0}
            d1 = digit_candidates[1][rank] if rank < len(digit_candidates[1]) else {"digit":"0","dev":"0","proba":0.0}
            d2 = digit_candidates[2][rank] if rank < len(digit_candidates[2]) else {"digit":"0","dev":"0","proba":0.0}

        num_str = d0["digit"] + d1["digit"] + d2["digit"]
        avg_proba = (d0["proba"] + d1["proba"] + d2["proba"]) / 3.0
        dev_info = f"百:{d0['dev']} 十:{d1['dev']} 一:{d2['dev']}"
        predictions[types[rank]] = (num_str, avg_proba, dev_info)

    # next_w は prepare_ai_data が返した「直近の次回（weekday index 0-4）」です
    next_label = f"{weekday_labels[next_w]}（次回）" if isinstance(next_w, (int, np.integer)) else "次回"

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_text = f"=== AI予測日時 : {now_str} ({mode_text}モード) ===\n"
    log_text += f"対象: 次回（直近のデータの翌回: {next_label}） / 前回番号: {last_actual_number}\n"
    for title, (num, proba, dev) in predictions.items():
        log_text += f" {title} -> 【 {num} 】 (信頼度: {proba:.1f}% / {dev})\n"
    log_text += "\n"
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(log_text)

    return last_actual_number, next_label, predictions


# --- Streamlit UI ---
st.title("🔮 ナンバーズ3 AI予測システム（ルーレット版・アップロード対応）")
st.markdown("CSV をブラウザでエクスポートして手動でアップロードすると、内部フォーマットに変換して AI 予測（次回＝直近のデータの翌回）を実行します。")

uploaded_file = st.file_uploader("過去データ CSV をアップロード（回号, 抽せん日, 当せん番号）", type=["csv"])
if uploaded_file is not None:
    try:
        mode = process_uploaded_history_csv(uploaded_file)
        st.success("CSV を受け取り、内部形式に変換して保存しました。AI 予測を実行します。")
        last_num, next_day, preds = run_prediction(CSV_FILE, mode)
        st.subheader(f"📊 予測結果（{next_day}）")
        st.info(f"💡 前回（ベース）の番号: **{last_num}**")
        col1, col2, col3 = st.columns(3)
        for i, (title, (num, proba, dev)) in enumerate(preds.items()):
            target_col = col1 if i == 0 else col2 if i == 1 else col3
            with target_col:
                st.metric(label=title, value=num)
                st.caption(f"🤖 期待値: **{proba:.1f}%**")
                st.caption(f"_{dev}_")
    except Exception as e:
        st.error(f"CSV の読み込みまたは変換に失敗しました: {e}")

st.divider()
st.subheader("📜 過去の予測履歴ログ")
if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        history_data = f.read()
    st.text_area(label="predictions_history.txt の中身", value=history_data, height=200)
else:
    st.caption("まだ予測履歴はありません。CSV をアップロードすると自動で作成されます。")
