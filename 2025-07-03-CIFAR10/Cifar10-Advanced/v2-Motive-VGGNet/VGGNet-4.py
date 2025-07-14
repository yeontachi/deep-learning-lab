# Data Augmentation and Early Stopping with VGGNet-like CNN on CIFAR-10
# RandomCrop, RandomHorizontalFlip, and Early Stopping Implementation

from tqdm import tqdm  
import ssl
import torch
import torch.nn as nn
from torchvision import transforms, datasets

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

original_dataset = CIFAR10(root="./data/", 
                           train=True,
                           download=True,
                           transform=original_transform)
augmented_dataset = CIFAR10(root="./data/",
                            train=True,
                            download=True,
                            transform=augmented_transform)

# 데이터셋 합치기
combined_dataset = ConcatDataset([original_dataset, augmented_dataset])

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
model = CNN().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

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
    
    early_stopping(test_loss)
    if early_stopping.early_stop:
        print("Early stopping triggered. Stopping training.")
        break
