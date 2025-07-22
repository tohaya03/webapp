# このセルで書かれたコードは「app.py」というPythonファイルに保存されます

# === 必要なライブラリのインポート ===
import streamlit as st  # Webアプリを簡単に作れるフレームワーク
import matplotlib.pyplot as plt  # グラフ描画ライブラリ（円グラフに使用）
from PIL import Image  # 画像の読み込み・処理ライブラリ
from model import predict  # 自作モデルの予測関数（model.py からインポート）

# === StreamlitアプリのUI構築 ===
# サイドバーにタイトルを表示
st.sidebar.title("画像認識アプリ")

# サイドバーに説明文を追加
st.sidebar.write("オリジナルの画像認識モデルを使って何の画像かを判定します。")

# 空行追加（UIの見た目を整える）
st.sidebar.write("")

# === 入力画像の取得方法を選択 ===
# ラジオボタンで「画像アップロード」か「カメラ撮影」を選ぶ
img_source = st.sidebar.radio("画像のソースを選択してください。",
                              ("画像をアップロード", "カメラで撮影"))

# ユーザーが選んだ方法に応じて、画像を取得
if img_source == "画像をアップロード":
    img_file = st.sidebar.file_uploader("画像を選択してください。", type=["png", "jpg"])
elif img_source == "カメラで撮影":
    img_file = st.camera_input("カメラで撮影")

# === 画像が選ばれた場合の処理 ===
if img_file is not None:
    # スピナー表示（処理中をユーザーに知らせる）
    with st.spinner("推定中..."):
        img = Image.open(img_file)  # 画像を開く
        st.image(img, caption="対象の画像", width=480)  # 表示（キャプション付き）
        st.write("")  # 空行

        # === モデルによる予測実行 ===
        results = predict(img)  # model.py の predict 関数を使って予測

        # === 結果のテキスト表示 ===
        st.subheader("判定結果")  # セクション見出し
        n_top = 3  # 確率の高い上位3件を表示
        for result in results[:n_top]:
            st.write(str(round(result[2]*100, 2)) + "%の確率で" + result[0] + "です。")

        # === 結果の円グラフ表示 ===
        pie_labels = [result[1] for result in results[:n_top]]  # 英語ラベル
        pie_labels.append("others")  # その他カテゴリ
        pie_probs = [result[2] for result in results[:n_top]]  # 上位の確率
        pie_probs.append(sum([result[2] for result in results[n_top:]]))  # 残りの合計

        # 円グラフのプロパティ設定と描画
        fig, ax = plt.subplots()
        wedgeprops = {"width": 0.3, "edgecolor": "white"}  # ドーナツ型にする設定
        textprops = {"fontsize": 6}  # 文字サイズ小さめ
        ax.pie(pie_probs, labels=pie_labels, counterclock=False, startangle=90,
               textprops=textprops, autopct="%.2f", wedgeprops=wedgeprops)

        # グラフをStreamlit上に表示
        st.pyplot(fig)
