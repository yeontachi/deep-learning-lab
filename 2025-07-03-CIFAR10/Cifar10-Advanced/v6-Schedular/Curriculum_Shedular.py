from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
import numpy as np

# 1. 디바이스 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

BATCH_SIZE = 32
STAGE_EPOCHS = 10
FINE_TUNE_EPOCHS = 5

# CIFAR-10 정규화 기준
mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)

# Transform 정의
original_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
augmented_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Step 1: Pretrain 간단 모델로 confidence 추정
raw_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transforms.ToTensor())
pre_model = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.Conv2d(32, 32, 3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),

    nn.Conv2d(32, 64, 3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.Conv2d(64, 64, 3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.Conv2d(64, 64, 3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),

    nn.Flatten(),
    nn.Linear(8 * 8 * 64, 1024),
    nn.BatchNorm1d(1024),
    nn.ReLU(),
    nn.Linear(1024, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Linear(128, 10),
    nn.LogSoftmax(dim=1)
).to(DEVICE)


optimizer = torch.optim.Adam(pre_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
pre_loader = DataLoader(raw_dataset, batch_size=128, shuffle=True)

pre_model.train()
for images, labels in pre_loader:
    images, labels = images.to(DEVICE), labels.to(DEVICE)
    optimizer.zero_grad()
    outputs = pre_model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    break  # 1 배치만 학습

# Step 2: Confidence 점수 계산
pre_model.eval()
confidence_scores = []
with torch.no_grad():
    for image, _ in raw_dataset:
        image = image.unsqueeze(0).to(DEVICE)
        prob = torch.softmax(pre_model(image), dim=1)
        confidence_scores.append(torch.max(prob).item())

confidence_scores = np.array(confidence_scores)
sorted_indices = np.argsort(-confidence_scores)

# Step 3: 데이터 분할
num_total = len(sorted_indices)
stage1_idx = sorted_indices[:int(0.3 * num_total)]
stage2_idx = sorted_indices[int(0.3 * num_total):int(0.7 * num_total)]
stage3_idx = sorted_indices[int(0.7 * num_total):]

original_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=original_transform)
augmented_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=augmented_transform)

stage1_loader = DataLoader(Subset(original_dataset, stage1_idx), batch_size=BATCH_SIZE, shuffle=True)
stage2_loader = DataLoader(Subset(augmented_dataset, stage2_idx), batch_size=BATCH_SIZE, shuffle=True)
stage3_loader = DataLoader(Subset(augmented_dataset, stage3_idx), batch_size=BATCH_SIZE, shuffle=True)

test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Step 4: CNN 정의
import torch.nn.functional as F

# ✅ Residual Block 정의
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

# ✅ 전체 CNN 모델 정의
class ResidualCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = ResidualBlock(3, 32)           # 32x32 -> 32x32
        self.block2 = ResidualBlock(32, 32)          # 32x32 -> 32x32
        self.pool1 = nn.MaxPool2d(2, 2)              # 32x32 -> 16x16

        self.block3 = ResidualBlock(32, 64)          # 16x16 -> 16x16
        self.block4 = ResidualBlock(64, 64)          # 16x16 -> 16x16
        self.block5 = ResidualBlock(64, 64)          # 16x16 -> 16x16
        self.pool2 = nn.MaxPool2d(2, 2)              # 16x16 -> 8x8

        self.fc1 = nn.Linear(8 * 8 * 64, 1024)
        self.bn_fc1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool1(x)

        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.pool2(x)

        x = x.view(-1, 8 * 8 * 64)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# Step 5: 학습 함수
def train(model, train_loader, optimizer, epoch):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for batch_idx, (image, label) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
        image, label = image.to(DEVICE), label.to(DEVICE)

        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(output, 1)
        correct += predicted.eq(label).sum().item()
        total += label.size(0)
    return running_loss / len(train_loader), 100. * correct / total

def evaluate(model, test_loader):
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for image, label in test_loader:
            image, label = image.to(DEVICE), label.to(DEVICE)
            output = model(image)
            test_loss += criterion(output, label).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(label.view_as(pred)).sum().item()
    return test_loss / len(test_loader), 100. * correct / len(test_loader.dataset)

# 5. 모델 생성 및 학습
model = ResidualCNN().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.NLLLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=0.5, patience=3,
                                                       )
#early_stopping = EarlyStopping(patience=5, min_delta=0.001)

for stage_num, loader in enumerate([stage1_loader, stage2_loader, stage3_loader], 1):
    print(f"\n Starting Stage {stage_num} Training")
    for epoch in range(1, STAGE_EPOCHS + 1):
        train_loss, train_acc = train(model, loader, optimizer, epoch)
        test_loss, test_acc = evaluate(model, test_loader)

        print(f"[Stage {stage_num} | Epoch {epoch}] Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | Test Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")
        scheduler.step(test_loss)
       # early_stopping(test_loss)
       # if early_stopping.early_stop:
        #    break

# Fine-tuning
from torch.utils.data import ConcatDataset

print("\n🔧 Fine-tuning on full dataset (Stage 4: 원본 + 증강 = 100,000장)")

combined_dataset = ConcatDataset([original_dataset, augmented_dataset])
combined_loader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, shuffle=True)

FINE_TUNE_EPOCHS = 10

# (선택) EarlyStopping 등 추가 가능
# fine_tune_early_stopping = EarlyStopping(patience=5, min_delta=0.001)

for epoch in range(1, FINE_TUNE_EPOCHS + 1):
    train_loss, train_acc = train(model, combined_loader, optimizer, epoch)
    test_loss, test_acc = evaluate(model, test_loader)

    print(f"[Fine-Tune | Epoch {epoch}] Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | Test Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")
    scheduler.step(test_loss)
    # EarlyStopping 조건 사용 시
    # fine_tune_early_stopping(test_loss)
    # if fine_tune_early_stopping.early_stop:
    #     break