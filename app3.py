import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from geomdl import NURBS, knotvector
import matplotlib.image as mpimg
import pandas as pd
import json
from datetime import datetime, timedelta
import gspread
from google.oauth2.service_account import Credentials

# 認証スコープ設定
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
st.title("🚗 NURBS Car Silhouette Editor ")
st.markdown("""
本アンケートは、**早稲田大学の研究プロジェクト**の一環として実施しているものです。  
「**言葉によるエンジニアリング**」というテーマのもと、**言葉から理想的な自動車の形状を導出すること**を目的としています。  
本アンケートでは、参加者の皆さまの操作結果をもとに、**言葉と形状の関係性**を分析いたします。  
なお、回答内容から個人を特定することは一切ありません。

実施者：早稲田大学 情報生産システム研究科 荒川研究室 尾﨑椋太

---

⚠️ **操作はPC（パソコン）でのご利用を推奨しています。**  
スマートフォンやタブレットからでも操作可能ですが、スライダーの操作がしづらい場合があります。

---

## 操作方法

1. 左のサイドバーで **車種を選択** してください。  
2. 各 **Point X** スライダーで点を左右に、**Point Y** スライダーで上下に動かすことができます。  
3. 車の先端を丸くしたり尖らせたりしたい場合は、**Weight（重み）** を調整してください。  
4. 基本的には **Weight を好みに調整** し、必要に応じて Point を微調整すると自然な形になります。  
5. 調整後、**透明度スライダー** で車体を黒くし、その印象に合う言葉を選んで評価してください。  
6. **複数の車種を回答する場合**は、1つの車種が終わったら **「保存」ボタンを押し、ページを更新** して次の車種を選んでください。  
7. 回答は何度でも行うことができます。

---
""")

# 車種データ
CAR_MODELS = {
    "軽自動車": {"ctrlpts": [[-0.5, 0], [-0.5, 2.0], [-0.2, 2.65], [1.5, 3.0], [2.6, 4.75],
                          [3.5, 5.1], [6.5, 5.1], [9.2, 5.1], [9.8, 4.5],
                          [9.88, 1.75], [10.1, 1.58], [10.0, 0]],
              "weights": [1.0, 2.0, 2.0, 5.0, 5.0, 2.0, 1.0, 7.0, 12.5, 1.0, 1.0, 1.0],
              "tire_coords": [(0.85, 0.1), (8.5, 0.1)],
              "ground_line": [-0.5, 10.0, 0.0],
              "bg_image": "Kei_car.jpg"},
    "コンパクトカー": {"ctrlpts": [[-0.6, -0.2], [-0.8, 2.0], [0.6, 3.2], [1.9, 3.4],
                              [3.8, 4.6], [6.6, 4.9], [10.0, 4.6],
                              [9.8, 3.9], [10.3, 2.0], [10.6, 1.2], [10.3, -0.2]],
                  "weights": [1.0, 6.0, 3.0, 5.0, 5.0, 6.0, 5.0, 3.0, 2.0, 1.0, 1.0],
                  "tire_coords": [(0.85, -0.2), (8.8, -0.2)],
                  "ground_line": [-0.6, 10.3, -0.2],
                  "bg_image": "compact_car.jpg"},
    "SUV": {"ctrlpts": [[-0.1, -0.5], [-0.15, 1.8], [0.8, 2.3], [2.8, 2.7],
                     [4.4, 4.2], [7.0, 4.45], [9.7, 4.0], [9.35, 3.4],
                     [10.0, 2.4], [10.0, 0.2], [9.8, -0.6], [9.2, -0.5]],
            "weights": [1.0, 5.0, 1.8, 4.4, 10.0, 6.0, 15.0, 18.5, 28.8, 28.8, 22.5, 10.0],
            "tire_coords": [(1.8, -0.5), (8.1, -0.5)],
            "ground_line": [-0.1, 9.2, -0.5],
            "bg_image": "SUV.jpg"}
}

# === サイドバー ===
selected_model = st.sidebar.selectbox("車種を選択", list(CAR_MODELS.keys()))
model_data = CAR_MODELS[selected_model]
initial_ctrlpts = model_data["ctrlpts"]
initial_weights = model_data["weights"]

# 透明度スライダー
st.sidebar.markdown("### ⚙️ 制御点と重み調整")
if st.sidebar.button("🔄 初期値にリセット"):
    for i, (pt, w) in enumerate(zip(initial_ctrlpts, initial_weights)):
        st.session_state[f"{selected_model}_x_{i}"] = float(pt[0])
        st.session_state[f"{selected_model}_y_{i}"] = float(pt[1])
        st.session_state[f"{selected_model}_w_{i}"] = float(w)
    st.rerun()

alpha = st.sidebar.slider("塗りつぶしの不透明度", 0.0, 1.0, 0.3, 0.05)

new_ctrlpts, new_weights = [], []
for i, (pt, w) in enumerate(zip(initial_ctrlpts, initial_weights)):
    st.sidebar.markdown(f"<small>Point {i}</small>", unsafe_allow_html=True)
    x_key, y_key, w_key = f"{selected_model}_x_{i}", f"{selected_model}_y_{i}", f"{selected_model}_w_{i}"
    if x_key not in st.session_state: st.session_state[x_key] = float(pt[0])
    if y_key not in st.session_state: st.session_state[y_key] = float(pt[1])
    if w_key not in st.session_state: st.session_state[w_key] = float(w)

    x = st.sidebar.slider(f"X", float(pt[0]-1), float(pt[0]+1), st.session_state[x_key], 0.1, key=x_key)
    y = st.sidebar.slider(f"Y", float(pt[1]-1), float(pt[1]+1), st.session_state[y_key], 0.1, key=y_key)
    ww = st.sidebar.slider(f"Weight", 0.1, 150.0, st.session_state[w_key], 0.1, key=w_key)

    new_ctrlpts.append([x, y])
    new_weights.append(ww)

# === NURBS曲線 ===
curve = NURBS.Curve()
curve.degree = 3
curve.ctrlpts = new_ctrlpts
curve.weights = new_weights
curve.knotvector = knotvector.generate(curve.degree, len(new_ctrlpts))
curve.evaluate()

# === 描画 ===
fig, ax = plt.subplots(figsize=(10, 7))
try:
    bg = mpimg.imread(model_data.get("bg_image", ""))
    ax.imshow(bg, extent=[-1, 11, -1.5, 6], aspect='auto', alpha=0.2)
except Exception:
    pass

for t in model_data.get("tire_coords", []):
    ax.add_patch(Circle((t[0], t[1]), 0.9, color='black'))

x0, x1, y_ground = model_data["ground_line"]
ax.plot([x0, x1], [y_ground, y_ground], '-', color='black', linewidth=1)

curve_pts = np.array(curve.evalpts)
ax.plot(curve_pts[:, 0], curve_pts[:, 1], color='blue', linewidth=2)
ctrl_np = np.array(new_ctrlpts)
ax.plot(ctrl_np[:, 0], ctrl_np[:, 1], '--', color='tab:red', marker='o')
ax.add_patch(Polygon(curve.evalpts + [new_ctrlpts[-1], new_ctrlpts[0]], closed=True, color='black', alpha=alpha))
ax.set_xlim(-3, 13)
ax.set_ylim(-3, 8)
ax.set_aspect('equal')
st.pyplot(fig)

# === 印象入力 ===
st.markdown("---")
st.markdown("### ✏️ この車の印象を教えてください")
adjective = st.selectbox("あなたの作った車を一言で表すと？",
    ["かわいい", "かっこいい", "頑丈そう", "速そう", "高級な", "親しみのある"]
)

# === 保存関数 ===
SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/1-mgxO9tqejwKehnbLS5B2JhCocdHH_xDWSZRLGKAE3A/edit?usp=sharing"

def save_to_google_sheet(model, ctrlpts, weights, alpha_value, adjective):
    try:
        credentials_info = dict(st.secrets["credentials_json"])
        creds = Credentials.from_service_account_info(credentials_info, scopes=scope)
        client = gspread.authorize(creds)
        worksheet = client.open_by_url(SPREADSHEET_URL).sheet1
        jst_time = datetime.utcnow() + timedelta(hours=9)
        row = [jst_time.strftime("%Y-%m-%d %H:%M:%S"), model,
        json.dumps(ctrlpts, ensure_ascii=False),
        json.dumps(weights, ensure_ascii=False),
        alpha_value, adjective]
        worksheet.append_row(row, value_input_option="USER_ENTERED")
        return True, None
    except Exception as e:
        return False, str(e)

if st.button("💾 保存する"):
    ok, err = save_to_google_sheet(selected_model, new_ctrlpts, new_weights, alpha, adjective)
    if ok:
        st.success("✅ Googleスプレッドシートに保存しました！")
    else:
        st.error("❌ 保存に失敗しました。")
        with st.expander("エラー内容を表示"):
            st.code(err)





