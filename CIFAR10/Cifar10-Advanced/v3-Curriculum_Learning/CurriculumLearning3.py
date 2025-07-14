import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# 디바이스 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CIFAR-10 정규화 값
mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)

# ==================== Transform 정의 ==================== #
class EdgeTransform:
    def __call__(self, img):
        img = np.array(img)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edges = Image.fromarray(edges).convert("L")
        return transforms.ToTensor()(edges).expand(3, -1, -1)

class FilledObjectTransform:
    def __call__(self, img):
        img_np = np.array(img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, contours, -1, 255, thickness=-1)
        mask_rgb = cv2.merge([mask]*3)
        filled = cv2.bitwise_and(img_np, mask_rgb)
        return transforms.ToTensor()(Image.fromarray(filled))

class ObjectColor_BackgroundGrayTransform:
    def __call__(self, img):
        img_np = np.array(img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, contours, -1, 255, thickness=-1)
        mask_3ch = cv2.merge([mask]*3)
        object_region = cv2.bitwise_and(img_np, mask_3ch)
        background_gray = cv2.GaussianBlur(gray, (7, 7), 0)
        background_rgb = cv2.merge([background_gray]*3)
        inverted_mask = cv2.bitwise_not(mask_3ch)
        background_region = cv2.bitwise_and(background_rgb, inverted_mask)
        blended = cv2.add(object_region, background_region)
        return transforms.ToTensor()(Image.fromarray(blended))

original_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# ==================== 모델 정의 ==================== #
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.conv1_3 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn1_3 = nn.BatchNorm2d(32)

        self.conv2_1 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.conv2_3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2_3 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(8 * 8 * 64, 1024)
        self.bn_fc1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.bn1_1(self.conv1_1(x)))
        x = torch.relu(self.bn1_2(self.conv1_2(x)))
        x = torch.relu(self.bn1_3(self.conv1_3(x)))
        x = self.pool(x)

        x = torch.relu(self.bn2_1(self.conv2_1(x)))
        x = torch.relu(self.bn2_2(self.conv2_2(x)))
        x = torch.relu(self.bn2_3(self.conv2_3(x)))
        x = self.pool(x)

        x = x.view(-1, 8 * 8 * 64)
        x = torch.relu(self.bn_fc1(self.fc1(x)))
        x = torch.relu(self.bn_fc2(self.fc2(x)))
        x = self.fc3(x)
        return torch.log_softmax(x, dim=1)

# ==================== 시각화 함수 ==================== #
def visualize_filters(model):
    filters = model.conv1_1.weight.data.clone().cpu()
    fig, axs = plt.subplots(4, 8, figsize=(12, 6))
    for i in range(32):
        ax = axs[i // 8, i % 8]
        img = filters[i].permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min())
        ax.imshow(img)
        ax.axis('off')
    plt.suptitle("Conv1_1 Filters")
    plt.tight_layout()
    plt.show()

def visualize_activations(model, sample_img):
    model.eval()
    with torch.no_grad():
        x = sample_img.unsqueeze(0).to(DEVICE)
        x = torch.relu(model.bn1_1(model.conv1_1(x)))
        activations = x[0].cpu()
        fig, axs = plt.subplots(4, 8, figsize=(12, 6))
        for i in range(32):
            ax = axs[i // 8, i % 8]
            ax.imshow(activations[i], cmap='gray')
            ax.axis('off')
        plt.suptitle("Conv1_1 Activation Maps")
        plt.tight_layout()
        plt.show()

# ==================== EarlyStopping ==================== #
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print("Early stopping triggered.")

# ==================== 학습 & 평가 함수 ==================== #
def train(model, loader, optimizer, criterion, epoch):
    model.train()
    correct, total, running_loss = 0, 0, 0
    for i, (x, y) in enumerate(tqdm(loader, desc=f"Epoch {epoch}")):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return running_loss / len(loader), 100 * correct / total

def evaluate(model, loader, criterion):
    model.eval()
    correct, total, test_loss = 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            test_loss += criterion(out, y).item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return test_loss / len(loader), 100 * correct / total

# ==================== 커리큘럼 학습 루프 ==================== #
# curriculum 설정
curriculum = [
    ("Stage 1: Edge Only", EdgeTransform(), 15),  # 윤곽선 학습 강조
    ("Stage 2: Filled Object Only", FilledObjectTransform(), 8),
    ("Stage 3: Object Color + Background Blurred", ObjectColor_BackgroundGrayTransform(), 10),
    ("Stage 4: Original", original_transform, 15),
]

# CIFAR-10 테스트셋 로드
test_dataset = CIFAR10(root="./data", train=False, download=True, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 모델, 옵티마이저, 손실함수 초기화
model = CNN().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 커리큘럼 학습
for stage_idx, (stage_name, transform, epochs) in enumerate(curriculum):
    print(f"\n🧠 Starting {stage_name} for {epochs} epochs")
    train_dataset = CIFAR10(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    use_early_stopping = (stage_idx != 0)  # Stage 1 제외
    early_stopping = EarlyStopping(patience=3) if use_early_stopping else None
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, epoch)
        test_loss, test_acc = evaluate(model, test_loader, criterion)

        print(f"[{stage_name} | Epoch {epoch}] Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
        early_stopping(test_loss)
        if early_stopping.early_stop:
            break

    # Stage 1 종료 직후 필터/activation 시각화
    if stage_idx == 0:
        sample_img, _ = next(iter(train_loader))
        visualize_filters(model)
        visualize_activations(model, sample_img[0])
