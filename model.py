import torch.nn as nn
import torch.nn.functional as F

# モデルの定義
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 畳み込み層（画像特徴を抽出）
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 入力: RGB画像
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # 2x2の領域を圧縮
        # 全結合層（特徴からクラスを分類）
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)  # 10クラス（CIFAR-10）

    def forward(self, x):
        # 順伝播（データを通して予測）
        x = F.relu(self.conv1(x))  # 1層目の畳み込み
        x = self.pool(F.relu(self.conv2(x)))  # 2層目の畳み込みとプーリング
        x = x.view(-1, 64 * 8 * 8)  # フラット化（全結合層への準備）
        x = F.relu(self.fc1(x))  # 1層目の全結合
        x = F.relu(self.fc2(x))  # 2層目の全結合
        x = self.fc3(x)  # 出力層
        return x

# モデルのインスタンスを作成
model = SimpleCNN()

# モデルの構造を確認
print(model)
