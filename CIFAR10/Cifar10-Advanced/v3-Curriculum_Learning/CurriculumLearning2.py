# reply 기법 적용
'''
이전 단계의 데이터 일부를 현재 단계의 학습에 함께 혼합하여,
모델이 이전 학습을 망각하지 않도록 도와주는 커리큘럼 학습 보완 기법
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import CIFAR10
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

# ==================== 디바이스 ==================== #
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== 정규화 ==================== #
mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)

# ==================== Transform 단계별 정의 ==================== #
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
test_transform = original_transform

# ==================== 모델 정의 (네가 만든 CNN) ==================== #
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

# ==================== EarlyStopping ==================== #
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001):
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

# ==================== 커리큘럼 + Replay 구조 ==================== #
transforms_by_stage = [
    EdgeTransform(),
    FilledObjectTransform(),
    ObjectColor_BackgroundGrayTransform(),
    original_transform
]

stage_names = [
    "Stage 1: Edge Only",
    "Stage 2: Filled Object Only",
    "Stage 3: Object Color + Background Blurred",
    "Stage 4: Original"
]

epochs_by_stage = [5, 8, 10, 15]
replay_ratio = 0.1  # 이전 단계 데이터 10% 사용

# 테스트셋 고정
test_dataset = CIFAR10(root="./data", train=False, download=True, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = CNN().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
prev_datasets = []

for stage_idx, (stage_name, transform, epochs) in enumerate(zip(stage_names, transforms_by_stage, epochs_by_stage)):
    print(f"\n🧠 Starting {stage_name} with Replay for {epochs} epochs")

    current_dataset = CIFAR10(root="./data", train=True, download=True, transform=transform)

    if stage_idx == 0:
        combined_dataset = current_dataset
    else:
        prev_subset_size = int(len(current_dataset) * replay_ratio)
        prev_subset = torch.utils.data.Subset(prev_datasets[-1], range(prev_subset_size))
        combined_dataset = ConcatDataset([current_dataset, prev_subset])

    prev_datasets.append(current_dataset)  # 현재 데이터셋 저장 (다음 스텝에서 replay용)

    train_loader = DataLoader(combined_dataset, batch_size=64, shuffle=True)
    early_stopping = EarlyStopping()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, epoch)
        test_loss, test_acc = evaluate(model, test_loader, criterion)

        print(f"[{stage_name} | Epoch {epoch}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

        early_stopping(test_loss)
        if early_stopping.early_stop:
            break
