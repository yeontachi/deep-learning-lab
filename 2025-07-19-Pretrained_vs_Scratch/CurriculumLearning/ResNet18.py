import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm

# 디바이스 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

BATCH_SIZE = 128
STAGE_EPOCHS = 20
FINE_TUNE_EPOCHS = 20

mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)

# Transform
plain_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
aug_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# ---------- Step 1: Confidence Pre-model ----------
pre_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transforms.ToTensor())

pre_model = nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(64, 10)
).to(DEVICE)

pre_loader = DataLoader(pre_dataset, batch_size=128, shuffle=True)
optimizer = torch.optim.Adam(pre_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

pre_model.train()
for images, labels in pre_loader:
    images, labels = images.to(DEVICE), labels.to(DEVICE)
    optimizer.zero_grad()
    outputs = pre_model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    break  # 1배치만 학습

# ---------- Step 2: Confidence 점수 계산 ----------
pre_model.eval()
conf_scores = []
with torch.no_grad():
    for img, _ in pre_dataset:
        img = img.unsqueeze(0).to(DEVICE)
        prob = torch.softmax(pre_model(img), dim=1)
        conf_scores.append(torch.max(prob).item())
conf_scores = np.array(conf_scores)
sorted_idx = np.argsort(-conf_scores)

import matplotlib.pyplot as plt
import numpy as np

def plot_confidence_histogram(confidence_scores, bins=50):
    """
    confidence_scores: numpy 배열 (e.g., np.array of max softmax values per sample)
    bins: 히스토그램 구간 수
    """
    plt.figure(figsize=(8, 5))
    plt.hist(confidence_scores, bins=bins, color='skyblue', edgecolor='black')
    plt.title("Confidence Score Distribution")
    plt.xlabel("Confidence (max softmax)")
    plt.ylabel("Number of Samples")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 예시: confidence_scores가 numpy 배열이라면 이렇게 사용
# plot_confidence_histogram(confidence_scores)

confidence_scores = np.array(conf_scores)  # 이미 존재한다고 가정
plot_confidence_histogram(conf_scores)


# ---------- Step 3: 데이터 분할 ----------
N = len(sorted_idx)
stage1_idx = sorted_idx[:int(0.3 * N)]
stage2_idx = sorted_idx[int(0.3 * N):int(0.7 * N)]
stage3_idx = sorted_idx[int(0.7 * N):]

plain_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=plain_transform)
aug_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=aug_transform)

stage1_loader = DataLoader(Subset(plain_dataset, stage1_idx), batch_size=BATCH_SIZE, shuffle=True)
stage2_loader = DataLoader(Subset(aug_dataset, stage2_idx), batch_size=BATCH_SIZE, shuffle=True)
stage3_loader = DataLoader(Subset(aug_dataset, stage3_idx), batch_size=BATCH_SIZE, shuffle=True)

test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=plain_transform)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# ---------- Step 4: ResNet-26 정의 ----------
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample:
            identity = self.downsample(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += identity
        return self.relu(x)

class ResNet26(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_channels = 64
        self.conv = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        self.layer1 = self._make_layer(64, 4)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 4, stride=2)
        self.layer4 = self._make_layer(512, 4, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, blocks, stride=1):
        down = None
        if stride != 1 or self.in_channels != out_channels:
            down = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        layers = [ResNetBlock(self.in_channels, out_channels, stride, down)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResNetBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

# ---------- Step 5: 학습 및 평가 함수 ----------
def train(model, loader, optimizer, criterion, epoch):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for image, label in tqdm(loader, desc=f"Epoch {epoch}", leave=False):
        image, label = image.to(DEVICE), label.to(DEVICE)
        optimizer.zero_grad()
        out = model(image)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += pred.eq(label).sum().item()
        total += label.size(0)
    return total_loss / len(loader), 100. * correct / total

def evaluate(model, loader):
    model.eval()
    loss, correct = 0.0, 0
    with torch.no_grad():
        for image, label in loader:
            image, label = image.to(DEVICE), label.to(DEVICE)
            out = model(image)
            loss += criterion(out, label).item()
            pred = out.argmax(dim=1)
            correct += pred.eq(label).sum().item()
    return loss / len(loader), 100. * correct / len(loader.dataset)

# ---------- Step 6: 커리큘럼 학습 + 파인튜닝 ----------
model = ResNet26().to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

for stage, loader in enumerate([stage1_loader, stage2_loader, stage3_loader], 1):
    print(f"\n[ Stage {stage} Training ]")
    for epoch in range(1, STAGE_EPOCHS + 1):
        train_loss, train_acc = train(model, loader, optimizer, criterion, epoch)
        test_loss, test_acc = evaluate(model, test_loader)
        print(f"[Stage {stage} | Epoch {epoch}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

# Fine-tuning
full_dataset = datasets.CIFAR10(root="./data", train=True, transform=aug_transform)
full_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True)

print("\n[ Fine-Tuning All Data ]")
for epoch in range(1, FINE_TUNE_EPOCHS + 1):
    train_loss, train_acc = train(model, full_loader, optimizer, criterion, epoch)
    test_loss, test_acc = evaluate(model, test_loader)
    print(f"[FineTune | Epoch {epoch}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
