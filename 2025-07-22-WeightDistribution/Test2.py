# Residual VGG-Inspired CNN with CIFAR-10
# 테스트 데이터에대한 성능이 높은 모델과 학습으로 가중치 분포를 확인하기 위함
from tqdm import tqdm  
import ssl
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

# 1. 디바이스 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

BATCH_SIZE = 32
EPOCHS = 30

from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import ConcatDataset

# 평균과 표준 편차(CIFAR-10 기준)
mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)

# 원본 데이터 변환
original_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# 증강 데이터 변환(RandomCrop, RandomHorizontalFlip)
augmented_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# 색상 기반 증강(ColorJitter)
'''colorjitter_transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.5, saturation=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])'''

original_dataset = CIFAR10(root="./data/", 
                           train=True,
                           download=True,
                           transform=original_transform)
augmented_dataset = CIFAR10(root="./data/",
                            train=True,
                            download=True,
                            transform=augmented_transform)
'''colorjitter_dataset = CIFAR10(root="./data/",
                              train=True,
                              download=True,
                              transform=colorjitter_transform)'''

# 데이터셋 합치기
combined_dataset = ConcatDataset([original_dataset, augmented_dataset]) # add colorjitter_dataset if needed

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
test_dataset = CIFAR10(root="./data/", 
                       train=False,
                       download=True,
                       transform=test_transform)

train_loader = torch.utils.data.DataLoader(dataset=combined_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=BATCH_SIZE,
                                          shuffle=False)

# 3. CNN 모델 정의
import torch
import torch.nn as nn
import torch.nn.functional as F

# Residual Block 정의
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

# 전체 CNN 모델 정의
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

# Early Stopping
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
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
                print("Early Stopping triggered. No imporvement for", self.patience, "epochs.")

# 4. 학습/평가 함수
model = ResidualCNN().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.NLLLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=0.5, patience=3,
                                                       )

def train(model, train_loader, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
    for batch_idx, (image, label) in progress_bar:
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

        if batch_idx % 50 == 0:
            progress_bar.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for image, label in test_loader:
            image, label = image.to(DEVICE), label.to(DEVICE)
            output = model(image)
            test_loss += criterion(output, label).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(label.view_as(pred)).sum().item()

    avg_test_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    return avg_test_loss, accuracy

# 5. 학습 루프
early_stopping = EarlyStopping(patience=5, min_delta=0.001)

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train(model, train_loader, optimizer, epoch)
    test_loss, test_acc = evaluate(model, test_loader)
    
    print(f"\n[EPOCH: {epoch}]")
    print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.2f}%")
    print(f"Test  Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%\n")

    # 스케줄러 step: test_loss 기준
    scheduler.step(test_loss)

    # EarlyStopping 검사
    early_stopping(test_loss)
    if early_stopping.early_stop:
        print("Early stopping triggered. Stopping training.")
        break

def plot_weight_histogram(model):
    for name, param in model.named_parameters():
        if 'weight' in name and param.requires_grad:
            plt.figure(figsize=(6, 4))
            plt.hist(param.data.cpu().numpy().flatten(), bins=100)
            plt.title(f'Weight Histogram for {name}')
            plt.xlabel('Weight Value')
            plt.ylabel('Frequenct')
            plt.grid()
            plt.show()
            
            
plot_weight_histogram(model)