import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from geomdl import NURBS, knotvector
import matplotlib.image as mpimg
import pandas as pd
import datetime
import os
import gspread
from google.oauth2.service_account import Credentials
import json

scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive"
]

# Streamlit Secrets から credentials_json を取得
credentials_info = dict(st.secrets["credentials_json"])

creds = Credentials.from_service_account_info(credentials_info, scopes=scope)
client = gspread.authorize(creds)


# ページ設定
st.set_page_config(page_title="NURBS Car Editor", layout="wide")
st.title("🚗 NURBS Car Silhouette Editor (Streamlit)")
st.markdown("左のサイドバーで車種を選び、制御点と重みを調整してください。")

# 自動翻訳を無効化（HTMLメタタグを挿入）
st.markdown(
    """
    <meta name="google" content="notranslate">
    <style>
        body { 
            -webkit-user-select: text; 
        }
    </style>
    """,
    unsafe_allow_html=True
)


# === 車種データ ===
CAR_MODELS = {
    "Kei car": {
        "ctrlpts": [[-0.5, 0], [-0.5, 2.0], [-0.2, 2.65], [1.5, 3.0], [2.6, 4.75], [3.5, 5.1], [6.5, 5.1], [9.2, 5.1], [9.8, 4.5], [9.88, 1.75], [10.1, 1.58], [10.0, 0]],
        "weights": [1.0, 2.0, 2.0, 5.0, 5.0, 2.0, 1.0, 7.0, 12.5, 1.0, 1.0, 1.0],
        "tire_coords": [(0.85, 0.1), (8.5, 0.1)],
        "ground_line": [-0.5, 10.0, 0.0],
        "bg_image": "kei_car.jpg"
    },
    "Compact": {
        "ctrlpts": [[-0.6, -0.2], [-0.8, 2.0], [0.6, 3.2], [1.9, 3.4], [3.8, 4.6], [6.6, 4.9], [10.0, 4.6], [9.8, 3.9], [10.3, 2.0], [10.6, 1.2], [10.3, -0.2]],
        "weights": [1.0, 6.0, 3.0, 5.0, 5.0, 6.0, 5.0, 3.0, 2.0, 1.0, 1.0],
        "tire_coords": [(0.85, -0.2), (8.8, -0.2)],
        "ground_line": [-0.6, 10.3, -0.2],
        "bg_image": "compact_car.jpg"
    },
    "SUV": {
        "ctrlpts": [[-0.1, -0.5], [-0.15, 1.8], [0.8, 2.3], [2.8, 2.7], [4.4, 4.2], [7.0, 4.45], [9.7, 4.0], [9.35, 3.4], [10.0, 2.4], [10.0, 0.2], [9.8, -0.6], [9.2, -0.5]],
        "weights": [1.0, 5.0, 1.8, 4.4, 10.0, 6.0, 15.0, 18.5, 28.8, 28.8, 22.5, 10.0],
        "tire_coords": [(1.8, -0.5), (8.1, -0.5)],
        "ground_line": [-0.1, 9.2, -0.5],
        "bg_image": "SUV.jpg"
    },
    "Sedan": {
        "ctrlpts": [[-0.4, 0.6], [-0.2, 2.1], [1.2, 2.8], [2.4, 2.9], [4.0, 4.0], [7.2, 4.0], [9.0, 3.1], [10.2, 3.0], [10.2, 2.2], [10.35, 1.6],[10.2, 0.6]],
        "weights": [1.0, 14.6, 20.2, 91.8, 100.0, 100.0, 100.0, 100.0, 14.3, 15.0, 1.0],
        "tire_coords": [(1.6, 1.0), (8.2, 1.0)],
        "ground_line": [-0.4, 10.2, 0.6],
        "bg_image": "sedan2.jpg"
    },
    "Minivan": {
        "ctrlpts": [[-0.5, 0], [-0.4, 2.0], [0, 2.5], [1.4, 2.9], [3.7, 5.0], [6.5, 5.0], [10.1, 5.0], [9.8, 4.6], [10.2,2.9], [10.1, 1.5], [10.1, 0]],
        "weights": [1.0, 9.1, 15.1, 30.2, 52.9, 15.1, 56.2, 17.8, 12.0, 11.8, 1.0],
        "tire_coords": [(1.6, 0.2), (8.3, 0.2)],
        "ground_line": [-0.5, 10.1, 0],
        "bg_image": "Minivan.jpg"
    },
    "Coupe": {
        "ctrlpts": [[0, 0.8], [0.1, 2.25], [0.8, 2.7], [3.4, 3.2], [4.6, 3.85], [6.0, 4.0], [7.2, 3.7], [8.4, 3.5], [9.4, 3.0], [9.8, 2.0], [9.5, 0.8]],
        "weights": [1.0, 9.1, 15.1, 30.2, 52.9, 30.0, 56.2, 20.0, 17.8, 11.8, 1.0],
        "tire_coords": [(1.8, 1.0), (8.0, 1.0)],
        "ground_line": [0, 9.4, 0.8],
        "bg_image": "coope.jpg"
    }
}

# === サイドバー ===
selected_model = st.sidebar.selectbox("車種を選択", list(CAR_MODELS.keys()))
model_data = CAR_MODELS[selected_model]
initial_ctrlpts = model_data["ctrlpts"]
initial_weights = model_data.get("weights", [])

if len(initial_weights) < len(initial_ctrlpts):
    if len(initial_weights) == 0:
        initial_weights = [1.0] * len(initial_ctrlpts)
    else:
        initial_weights = initial_weights + [initial_weights[-1]] * (len(initial_ctrlpts) - len(initial_weights))
elif len(initial_weights) > len(initial_ctrlpts):
    initial_weights = initial_weights[:len(initial_ctrlpts)]

st.sidebar.markdown("### ⚙️ 制御点と重み調整")
alpha = st.sidebar.slider("塗りつぶしの不透明度", 0.0, 1.0, 0.3, 0.05)

new_ctrlpts, new_weights = [], []
for i, (pt, w) in enumerate(zip(initial_ctrlpts, initial_weights)):
    x_key, y_key, w_key = f"{selected_model}_x_{i}", f"{selected_model}_y_{i}", f"{selected_model}_w_{i}"
    if x_key not in st.session_state: st.session_state[x_key] = float(pt[0])
    if y_key not in st.session_state: st.session_state[y_key] = float(pt[1])
    if w_key not in st.session_state: st.session_state[w_key] = float(w)
    x = st.sidebar.slider(f"Point {i} X", float(pt[0]-1), float(pt[0]+1), st.session_state[x_key], 0.1, key=x_key)
    y = st.sidebar.slider(f"Point {i} Y", float(pt[1]-1), float(pt[1]+1), st.session_state[y_key], 0.1, key=y_key)
    ww = st.sidebar.slider(f"Weight {i}", 0.1, 150.0, st.session_state[w_key], 0.1, key=w_key)
    new_ctrlpts.append([float(x), float(y)])
    new_weights.append(float(ww))

if st.sidebar.button("初期値にリセット"):
    for i, (pt, w) in enumerate(zip(initial_ctrlpts, initial_weights)):
        st.session_state[f"{selected_model}_x_{i}"] = float(pt[0])
        st.session_state[f"{selected_model}_y_{i}"] = float(pt[1])
        st.session_state[f"{selected_model}_w_{i}"] = float(w)
    st.rerun()

# === NURBS曲線生成 ===
curve = NURBS.Curve()
curve.degree = 3
curve.ctrlpts = new_ctrlpts
curve.weights = new_weights
curve.knotvector = knotvector.generate(curve.degree, len(new_ctrlpts))
curve.delta = 0.01
curve.evaluate()

# === 描画 ===
fig, ax = plt.subplots(figsize=(10, 7))
try:
    bg = mpimg.imread(model_data.get("bg_image", ""))
    ax.imshow(bg, extent=[-1, 11, -1.5, 6], aspect='auto', alpha=0.2)
except Exception:
    pass

for t in model_data.get("tire_coords", []):
    ax.add_patch(Circle((t[0], t[1]), 0.9, color='black', zorder=1))

if "ground_line" in model_data:
    x0, x1, y_ground = model_data["ground_line"]
    ax.plot([x0, x1], [y_ground, y_ground], '-', color='black', linewidth=1)

curve_pts = np.array(curve.evalpts)
ax.plot(curve_pts[:, 0], curve_pts[:, 1], color='blue', linewidth=2)
ctrl_np = np.array(new_ctrlpts)
ax.plot(ctrl_np[:, 0], ctrl_np[:, 1], '--', color='tab:red', marker='o')
poly_pts = curve.evalpts + [new_ctrlpts[-1], new_ctrlpts[0]]
ax.add_patch(Polygon(poly_pts, closed=True, color='black', alpha=alpha))
ax.set_xlim(-3, 13)
ax.set_ylim(-3, 8)
ax.set_aspect('equal')
ax.grid(True)
st.pyplot(fig)

# === 形容詞入力欄（🔹追加） ===
st.markdown("---")
st.markdown("### ✏️ この車の印象を教えてください")
adjective = st.selectbox(
    "この車を一言で表すと？",
    ["かわいい", "かっこいい", "シンプル", "未来的", "高級感がある", "スポーティ", "落ち着いている"]
)

# === Google Sheets保存設定（Secrets を利用） ===
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/1-mgxO9tqejwKehnbLS5B2JhCocdHH_xDWSZRLGKAE3A/edit?usp=sharing"

def save_to_google_sheet(model, ctrlpts, weights, alpha_value, adjective):
    try:
        # 1) Streamlit Secrets から認証情報を取得（st.secrets に credentials_json セクションが入っていること）
        if "credentials_json" not in st.secrets:
            raise RuntimeError("Streamlit secrets に 'credentials_json' が見つかりません。Manage app → Secrets を確認してください。")

        credentials_info = dict(st.secrets["credentials_json"])
        creds = Credentials.from_service_account_info(credentials_info, scopes=SCOPES)
        client = gspread.authorize(creds)

        # 2) スプレッドシートに書き込み
        spreadsheet = client.open_by_url(SPREADSHEET_URL)
        worksheet = spreadsheet.sheet1

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ctrlpts_str = json.dumps(ctrlpts, ensure_ascii=False)
        weights_str = json.dumps(weights, ensure_ascii=False)

        row = [timestamp, model, ctrlpts_str, weights_str, alpha_value, adjective]
        # 文字化け回避
        row = [str(v).encode("utf-8", "ignore").decode("utf-8") for v in row]
        worksheet.append_row(row, value_input_option="USER_ENTERED")

        return True, None
    except Exception as e:
        return False, str(e)

# === 送信ボタン ===
if st.button("💾 保存する"):
    ok, err = save_to_google_sheet(selected_model, new_ctrlpts, new_weights, alpha, adjective)
    if ok:
        st.success("✅ Googleスプレッドシートに保存しました！")
    else:
        st.error("❌ 保存に失敗しました。")
        with st.expander("エラー内容を表示"):
            st.code(err, language="text")

# デバック用
st.write("secrets keys:", list(st.secrets.keys()))
st.write("has credentials_json?:", "credentials_json" in st.secrets)
# private_key の先頭30文字を表示（改行があるかを可視化）
if "credentials_json" in st.secrets:
    st.write(repr(st.secrets["credentials_json"]["private_key"][:60]))

