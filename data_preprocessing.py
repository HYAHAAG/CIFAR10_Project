import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# データの前処理
transform = transforms.Compose([
    transforms.ToTensor(),  # 画像をテンソル（数値配列）に変換
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 値を-1～1に正規化
])

# データセットの取得
train_data = torchvision.datasets.CIFAR10(
    root='E:/CIFAR10_Project/data', train=True, download=True, transform=transform)
test_data = torchvision.datasets.CIFAR10(
    root='E:/CIFAR10_Project/data', train=False, download=True, transform=transform)

# データローダの設定
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# データの確認
print(f"Train dataset size: {len(train_data)}")
print(f"Test dataset size: {len(test_data)}")
print(f"Example train data shape: {train_data[0][0].shape}")
