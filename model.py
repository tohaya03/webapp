# このセルで書かれたコードは「model.py」というPythonファイルに保存されます

# === ライブラリのインポート ===
# PyTorchの基本機能とニューラルネットワークの構築に必要なモジュール
import torch
import torch.nn as nn
import torch.nn.functional as F

# torchvisionは画像の前処理や学習済みモデルの提供を行うライブラリ
from torchvision import models, transforms

# PILは画像を読み込んだり、サイズを変更したりするためのライブラリ
from PIL import Image

# === CIFAR-10のラベル名（日本語と英語） ===
classes_ja = ["飛行機", "自動車", "鳥", "猫", "鹿", "犬", "カエル", "馬", "船", "トラック"]
classes_en = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
n_class = len(classes_ja)  # クラス数（=10）
img_size = 32  # 入力画像のサイズ（CIFAR-10の標準サイズ）

# === CNNのモデル定義 ===
class Net(nn.Module):
    def __init__(self):
        super().__init__()  # 親クラスの初期化
        self.conv1 = nn.Conv2d(3, 6, 5)  # 1層目：RGB画像 → 6チャンネル
        self.pool = nn.MaxPool2d(2, 2)  # プーリング層：画像サイズを半分に
        self.conv2 = nn.Conv2d(6, 16, 5)  # 2層目：6チャンネル → 16チャンネル
        self.fc1 = nn.Linear(16*5*5, 256)  # 全結合層1
        self.dropout = nn.Dropout(p=0.5)  # ドロップアウト：過学習防止
        self.fc2 = nn.Linear(256, 10)  # 全結合層2：10クラス分類

    def forward(self, x):  # 順伝播の処理
        x = self.pool(F.relu(self.conv1(x)))  # 畳み込み1 → ReLU → プーリング
        x = self.pool(F.relu(self.conv2(x)))  # 畳み込み2 → ReLU → プーリング
        x = x.view(-1, 16*5*5)  # Flatten（1次元に変換）
        x = F.relu(self.fc1(x))  # 全結合1 → ReLU
        x = self.dropout(x)  # ドロップアウト
        x = self.fc2(x)  # 出力層
        return x

# === 推論関数（画像を与えると結果を返す） ===
def predict(img):
    # === 入力画像の前処理 ===
    img = img.convert("RGB")  # RGB形式に変換（3チャンネルにする）
    img = img.resize((img_size, img_size))  # CIFAR-10に合わせて32×32にリサイズ

    # ToTensorでTensor化し、Normalizeで正規化（0〜1の範囲）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))  # 平均0、標準偏差1（簡易）
    ])
    img = transform(img)
    x = img.reshape(1, 3, img_size, img_size)  # バッチ次元を追加（1枚の画像）

    # === モデルの読み込み ===
    net = Net()  # モデルのインスタンス化
    net.load_state_dict(torch.load(
        "model_cnn.pth", map_location=torch.device("cpu")  # CPU環境で読み込み
    ))

    # === 推論実行 ===
    net.eval()  # 評価モードに切り替え（学習機能を無効化）
    y = net(x)  # 推論結果（スコア）

    # === 結果をクラスごとの確率に変換し、上位からソート ===
    y_prob = torch.nn.functional.softmax(torch.squeeze(y))  # softmaxで確率に変換
    sorted_prob, sorted_indices = torch.sort(y_prob, descending=True)  # 高い順に並べる

    # 上位から順に（日本語, 英語, 確率）で返す
    return [(classes_ja[idx], classes_en[idx], prob.item()) for idx, prob in zip(sorted_indices, sorted_prob)]
