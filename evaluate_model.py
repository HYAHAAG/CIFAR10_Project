import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import SimpleCNN  # モデルのインポート

# 1. データの準備
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_data = torchvision.datasets.CIFAR10(
    root='E:/CIFAR10_Project/data', train=False, download=True, transform=transform)

test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# 2. モデルの読み込み
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)

# モデルを評価モードに設定
model.load_state_dict(torch.load("E:/CIFAR10_Project/cifar10_model.pth", weights_only=True))
model.eval()

# 3. モデルの評価
correct = 0
total = 0
with torch.no_grad():  # 評価時は勾配を計算しない
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
