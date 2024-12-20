import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from model import SimpleCNN  # 先ほど作成したモデルをインポート

# 1. データの準備
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = torchvision.datasets.CIFAR10(
    root='E:/CIFAR10_Project/data', train=True, download=True, transform=transform)
test_data = torchvision.datasets.CIFAR10(
    root='E:/CIFAR10_Project/data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# 2. モデルの準備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)

# 3. 損失関数と最適化アルゴリズム
criterion = nn.CrossEntropyLoss()  # 損失関数（クロスエントロピー）
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 最適化アルゴリズム（Adam）

# 4. モデルの訓練
num_epochs = 10  # エポック数
for inputs, labels in train_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
    print(f"Output shape: {outputs.shape}")
    break  # 最初のバッチだけ確認

for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # 順伝播
        optimizer.zero_grad()  # 勾配の初期化
        outputs = model(inputs)  # モデルの予測
        loss = criterion(outputs, labels)  # 損失を計算
        loss.backward()  # 逆伝播で勾配を計算
        optimizer.step()  # パラメータの更新

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

# 5. モデルの保存
torch.save(model.state_dict(), "E:/CIFAR10_Project/cifar10_model.pth")
print("Model saved successfully.")
