import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.data import random_split, Subset
from collections import defaultdict
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

# 1. 디바이스 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# 2. 하이퍼파라미터
BATCH_SIZE = 32
EPOCHS = 10

# 3. 데이터셋 불러오기 및 분할
transform = transforms.ToTensor()

# 전체 학습 데이터셋 (train=True)
full_train_dataset = datasets.MNIST(root="../data/MNIST", train=True, download=True, transform=transform)

# 전체 테스트 데이터셋 (test=True)
full_test_dataset = datasets.MNIST(root="../data/MNIST", train=False, download=True, transform=transform)

# 3-1. 학습:검증 = 8:2 분할
total_size = len(full_train_dataset)  # 60000
val_size = int(total_size * 0.2)      # 12000
train_size = total_size - val_size    # 48000

train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

# 3-2. 클래스당 100장씩 추출하여 테스트셋 구성 (총 1000장)
label_to_indices = defaultdict(list)
for idx, (_, label) in enumerate(full_test_dataset):
    if len(label_to_indices[label]) < 100:
        label_to_indices[label].append(idx)

final_test_indices = [idx for indices in label_to_indices.values() for idx in indices]
final_test_dataset = Subset(full_test_dataset, final_test_indices)

# 3-3. DataLoader 구성
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
final_test_loader = torch.utils.data.DataLoader(final_test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 4. CNN 모델 정의
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=7, padding=3)
        
        self.fc1 = nn.Linear(128 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = x.view(-1, 128 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 5. 모델, 옵티마이저, 손실 함수
model = CNN().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 6. 학습 함수
def train(model, train_loader, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Train Epoch {epoch}")
    for batch_idx, (image, label) in pbar:
        image, label = image.to(DEVICE), label.to(DEVICE)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(label).sum().item()
        total += label.size(0)

        pbar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    acc = 100. * correct / total
    return avg_loss, acc


# 7. 평가 함수 (검증/테스트 공용)
def evaluate(model, loader):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for image, label in loader:
            image, label = image.to(DEVICE), label.to(DEVICE)
            output = model(image)
            total_loss += criterion(output, label).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(label).sum().item()

    avg_loss = total_loss / (len(loader.dataset) / BATCH_SIZE)
    accuracy = 100. * correct / len(loader.dataset)
    return avg_loss, accuracy

# Early stopping(에폭 수는 대략 20정도 예상)
'''class EarlyStopping:
    def __init__(self, patience=5, verbose=True):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.verbose = verbose

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"🟡 EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

early_stopping = EarlyStopping(patience=5)'''

# 8. 학습 루프
train_losses, train_accs = [], []
val_losses, val_accs = [], []

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train(model, train_loader, optimizer)
    val_loss, val_acc = evaluate(model, val_loader)

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(f"[EPOCH {epoch}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%\n")
    '''early_stopping(val_loss)
    if early_stopping.early_stop:
        print("Early stopping triggered. Stopping training.")
        break'''

# 🔹 8-1. 최종 평균 성능 출력
avg_train_loss = sum(train_losses) / len(train_losses)
avg_train_acc = sum(train_accs) / len(train_accs)
avg_val_loss = sum(val_losses) / len(val_losses)
avg_val_acc = sum(val_accs) / len(val_accs)

print("🔎 [Training Summary]")
print(f"Average Train Loss: {avg_train_loss:.4f}, Average Train Accuracy: {avg_train_acc:.2f}%")
print(f"Average Val   Loss: {avg_val_loss:.4f}, Average Val   Accuracy: {avg_val_acc:.2f}%\n")

# 9. 최종 평가
final_loss, final_acc = evaluate(model, final_test_loader)
print("Final Test Loss: {:.4f}".format(final_loss))
print("Final Test Accuracy: {:.2f}%".format(final_acc))

# 10. Confusion Matrix 시각화
def get_all_preds(model, loader):
    model.eval()
    all_preds = torch.tensor([], dtype=torch.long).to(DEVICE)
    all_labels = torch.tensor([], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        for image, label in loader:
            image, label = image.to(DEVICE), label.to(DEVICE)
            output = model(image)
            preds = output.argmax(dim=1)
            all_preds = torch.cat((all_preds, preds), dim=0)
            all_labels = torch.cat((all_labels, label), dim=0)

    return all_preds.cpu().numpy(), all_labels.cpu().numpy()

def plot_confusion_matrix(model, loader, class_names):
    y_pred, y_true = get_all_preds(model, loader)
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

class_names = [str(i) for i in range(10)]
plot_confusion_matrix(model, final_test_loader, class_names)
