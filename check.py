import pandas as pd
import json
import ast

CSV_FILE = "car_data.csv"

print("--- 診断を開始します ---")

try:
    # 先ほど成功したエンコーディングを指定してください
    # 方法1なら 'cp932', 方法2(メモ帳保存)なら 'utf-8-sig'
    df = pd.read_csv(CSV_FILE, encoding='cp932') 
    
    print(f"1. データ読み込み成功: 全 {len(df)} 行ありました。")
    
    if len(df) == 0:
        print("⚠️ エラー: データが0件です。CSVの中身が空か、正しく読み込めていません。")
    else:
        # 1行目のデータを取り出して確認
        row = df.iloc[0]
        print("\n2. 1行目のデータの確認:")
        # カラム名のリストを表示
        print(f"   カラム一覧: {list(df.columns)}")
        
        # 元のコードで指定していた列の中身を表示してみる
        # row[5] (制御点) と row[6] (重み) が本当にそこにあるか確認
        print(f"\n   [重要] 制御点データ(列5): {str(row.iloc[5])[:50]}...") # 長いので先頭50文字だけ
        print(f"   [重要] 重みデータ(列6):   {str(row.iloc[6])[:50]}...")

        # 解析テスト
        print("\n3. 解析テスト:")
        raw_ctrl = row.iloc[5]
        
        try:
            # JSONとして解析できるか？
            data = json.loads(raw_ctrl)
            print("   ✅ json.loads で解析できました！")
        except:
            print("   ❌ json.loads では解析できませんでした。")
            try:
                # Pythonのリストとして解析できるか？
                data = ast.literal_eval(raw_ctrl)
                print("   ✅ ast.literal_eval で解析できました！")
            except Exception as e:
                print(f"   ❌ ast.literal_eval も失敗しました: {e}")

except Exception as e:
    print(f"⚠️ 読み込み段階でエラーが発生しました: {e}")

print("--- 診断終了 ---")