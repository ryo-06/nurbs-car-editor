import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from geomdl import NURBS, knotvector
import json
import os
import ast
import re
import shutil

# === 設定 ===
CSV_FILE = "car_data.csv"
OUTPUT_DIR = "output_images"

# 保存先フォルダの作成
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# === 0. ヘルパー関数 ===
def find_keyword(text, keywords, default="unknown"):
    text_str = str(text).lower()
    for k in keywords:
        if k.lower() in text_str:
            return k
    return default

def get_age_label(text):
    match = re.search(r'(\d+s)', str(text))
    if match:
        return match.group(1)
    return "unknown"

def get_gender_label(text):
    text_str = str(text).upper()
    if "(M)" in text_str or "男性" in text_str: return "M"
    if "(F)" in text_str or "女性" in text_str: return "F"
    if "M" in text_str and "F" not in text_str: return "M"
    if "F" in text_str and "M" not in text_str: return "F"
    return "unknown"

# 形容詞リスト設定
ADJECTIVES = ["cute", "cool", "sturdy", "fast", "luxury", "familiar", "かわいい", "かっこいい", "頑丈そう", "速そう", "高級な", "親しみのある"]
ADJ_MAP = {
    "かわいい": "cute", "かっこいい": "cool", "頑丈そう": "sturdy",
    "速そう": "fast", "高級な": "luxury", "親しみのある": "familiar"
}
CAR_MODELS = {
    "軽自動車": { "tire": [(0.85, 0.1), (8.5, 0.1)] },
    "コンパクトカー": { "tire": [(0.85, -0.2), (8.8, -0.2)] },
    "SUV": { "tire": [(1.8, -0.5), (8.1, -0.5)] },
    "セダン": { "tire": [(1.6, 1.0), (8.2, 1.0)] },
    "ミニバン": { "tire": [(1.6, 0.2), (8.3, 0.2)] },
    "クーペ": { "tire": [(1.8, 1.0), (8.0, 1.0)] }
}

# === 1. データ修復と読み込み ===
print("--- [Step 1] データの読み込みと解析を開始します ---")

try:
    df_raw = pd.read_csv(CSV_FILE, header=None, encoding='utf-8', on_bad_lines='skip')
except:
    print("UTF-8での読み込みに失敗、cp932で試行します...")
    df_raw = pd.read_csv(CSV_FILE, header=None, encoding='cp932', on_bad_lines='skip')

cleaned_data = []

for index, row in df_raw.iterrows():
    try:
        row_list = [str(x) for x in row.tolist()]
        
        # 制御点データの位置を探す
        ctrl_idx = -1
        for i, val in enumerate(row_list):
            if val.strip().startswith('[['):
                ctrl_idx = i
                break
        
        if ctrl_idx == -1:
            continue

        # データ抽出
        gender_raw = row_list[ctrl_idx - 3] if ctrl_idx >= 3 else ""
        age_raw    = row_list[ctrl_idx - 2] if ctrl_idx >= 2 else ""
        model_raw  = row_list[ctrl_idx - 1]
        ctrl_raw   = row_list[ctrl_idx]
        weight_raw = row_list[ctrl_idx + 1]
        adj_raw    = row_list[ctrl_idx + 3] if len(row_list) > ctrl_idx + 3 else "unknown"
        timestamp  = row_list[0]
        
        # データクレンジング
        model_clean = "UnknownModel"
        m = model_raw
        if "Light" in m or "軽" in m: model_clean = "軽自動車"
        elif "Compact" in m or "コンパクト" in m: model_clean = "コンパクトカー"
        elif "SUV" in m: model_clean = "SUV"
        elif "Sedan" in m or "セダン" in m: model_clean = "セダン"
        elif "Minivan" in m or "ミニバン" in m: model_clean = "ミニバン"
        elif "Coupe" in m or "coupe" in m or "クー" in m: model_clean = "クーペ"

        found_adj = find_keyword(adj_raw, ADJECTIVES, default="unknown")
        if found_adj in ADJ_MAP:
            found_adj = ADJ_MAP[found_adj]
        
        found_age = get_age_label(age_raw)
        found_gender = get_gender_label(gender_raw)

        cleaned_data.append({
            "timestamp": timestamp,
            "gender": found_gender,
            "age": found_age,
            "model": model_clean,
            "adjective": found_adj,
            "ctrlpts": ctrl_raw,
            "weights": weight_raw,
            "idx": index
        })

    except Exception:
        pass

# === 2. 画像生成（差分更新） ===
print(f"--- [Step 2] 画像生成を開始します ({len(cleaned_data)}件) ---")
print("※ 作成済みの画像はスキップします")

count_gen = 0
count_skipped = 0

for row in cleaned_data:
    try:
        # 作成予定のファイル名
        filename = f"{row['idx']:03d}_{row['model']}_{row['age']}_{row['gender']}_{row['adjective']}.png"
        
        # チェック: 既にフォルダの中にあるか？ (cool/001_...png など)
        target_path_in_subfolder = os.path.join(OUTPUT_DIR, row['adjective'], filename)
        target_path_direct = os.path.join(OUTPUT_DIR, filename)
        
        # すでに作成済みならスキップ
        if os.path.exists(target_path_in_subfolder) or os.path.exists(target_path_direct):
            count_skipped += 1
            continue

        # === ここから画像生成 ===
        model_name = row['model']
        
        try:
            ctrlpts = json.loads(row['ctrlpts'])
            weights = json.loads(row['weights'])
        except:
            ctrlpts = ast.literal_eval(row['ctrlpts'])
            weights = ast.literal_eval(row['weights'])

        curve = NURBS.Curve()
        curve.degree = 3
        curve.ctrlpts = ctrlpts
        curve.weights = weights
        curve.knotvector = knotvector.generate(curve.degree, len(ctrlpts))
        curve.delta = 0.01
        curve.evaluate()

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.set_axis_off()

        tire_info = CAR_MODELS.get(model_name, {}).get("tire", [])
        for t in tire_info:
            ax.add_patch(Circle((t[0], t[1]), 0.9, color='black', zorder=1))

        poly_pts = curve.evalpts + [ctrlpts[-1], ctrlpts[0]]
        ax.add_patch(Polygon(poly_pts, closed=True, color='black', alpha=1.0))

        ax.set_aspect('equal')
        ax.set_xlim(-3, 13)
        ax.set_ylim(-3, 8)

        # 保存
        plt.savefig(target_path_direct, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        
        count_gen += 1
        if count_gen % 10 == 0:
            print(f"... 新規 {count_gen}枚 生成")

    except Exception as e:
        print(f"Error generating image for row {row['idx']}: {e}")

print(f"✅ 生成完了: 新規 {count_gen}枚 (スキップ {count_skipped}枚)")

# === 3. フォルダ整理 ===
print(f"--- [Step 3] フォルダ整理を開始します ---")

files = [f for f in os.listdir(OUTPUT_DIR) if f.lower().endswith('.png')]

if not files:
    print("整理対象の新規ファイルはありません。")
else:
    count_moved = 0
    for filename in files:
        try:
            name_without_ext = os.path.splitext(filename)[0]
            parts = name_without_ext.split('_')
            
            if len(parts) >= 2:
                adjective = parts[-1]
                
                folder_path = os.path.join(OUTPUT_DIR, adjective)
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                
                src_path = os.path.join(OUTPUT_DIR, filename)
                dst_path = os.path.join(folder_path, filename)
                
                # 移動先にファイルがあったら上書きのために一度消す（念のため）
                if os.path.exists(dst_path):
                    os.remove(dst_path)
                    
                shutil.move(src_path, dst_path)
                count_moved += 1
        except Exception as e:
            print(f"整理エラー ({filename}): {e}")

    print(f"✅ フォルダ整理完了: {count_moved}件移動")

print(f"\n=== 全工程が完了しました ===")