# CIFAR-10 데이터셋을 이용한 VGGNet 스타일의 CNN 모델 학습 코드
from tqdm import tqdm  # 추가 필요
import ssl
import torch
import torch.nn as nn
from torchvision import transforms, datasets

# 1. 디바이스 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

BATCH_SIZE = 32
EPOCHS = 30

# 2. 데이터셋 로드
train_dataset = datasets.CIFAR10(root="./data/",
                                 train=True,
                                 download=True,
                                 transform=transforms.ToTensor())
test_dataset = datasets.CIFAR10(root="./data/",
                                train=False,
                                download=True,
                                transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
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
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv1_3 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv2_1 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv2_3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(8 * 8 * 64, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = torch.relu(self.conv1_1(x))
        x = torch.relu(self.conv1_2(x))
        x = torch.relu(self.conv1_3(x))
        x = self.pool(x)
        
        x = torch.relu(self.conv2_1(x))
        x = torch.relu(self.conv2_2(x))
        x = torch.relu(self.conv2_3(x))
        x = self.pool(x)
        
        x = x.view(-1, 8 * 8 * 64)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.log_softmax(x, dim=1)

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
for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train(model, train_loader, optimizer, epoch)
    test_loss, test_acc = evaluate(model, test_loader)
    
    print(f"\n[EPOCH: {epoch}]")
    print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.2f}%")
    print(f"Test  Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%\n")
