import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 1. 디바이스 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU있다면 CUDA사용, 그렇지 않으면 CPU로 학습
print("Using device:", DEVICE)

# 2. 하이퍼파라미터
BATCH_SIZE = 32 # 한번에 학습할 데이터의 개수
EPOCHS = 30     # 전체 데이터셋을 10번 반복 학습

# 3. 데이터셋 불러오기
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(root="../data/MNIST", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="../data/MNIST", train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True) #매 epoch마다 데이터 순서를 무작위로 바꿈(shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 4. CNN 모델 정의
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # [B, 1, 28, 28] -> [B, 32, 28, 28]
        self.pool1 = nn.MaxPool2d(2, 2)                           # [B, 32, 28, 28] -> [B, 32, 14, 14]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # [B, 32, 14, 14] -> [B, 64, 14, 14]
        self.pool2 = nn.MaxPool2d(2, 2)                           # [B, 64, 14, 14] -> [B, 64, 7, 7]
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))     # Conv + ReLU
        x = self.pool1(x)             # MaxPooling
        x = F.relu(self.conv2(x))     # Conv + ReLU
        x = self.pool2(x)             # MaxPooling
        x = x.view(-1, 64 * 7 * 7)     # Flatten
        x = F.relu(self.fc1(x))       # FC + ReLU
        x = self.fc2(x)               # 출력층
        return x                      # CrossEntropyLoss에 raw logits 전달

# 5. 모델, 옵티마이저, 손실 함수
model = CNN().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 6. 학습 함수
def train(model, train_loader, optimizer, log_interval):
    model.train() #모델을 train() 모드로 설정
    
    #매 batch마다 forward, loss 계산, backward, optimizer 업데이트 수행
    for batch_idx, (image, label) in enumerate(train_loader):
        image, label = image.to(DEVICE), label.to(DEVICE)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0: # log_interval마다 로그 출력
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch, batch_idx * len(image), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

# 7. 평가 함수
def evaluate(model, test_loader):
    model.eval() # 모델을 eval() 모드로 설정
    test_loss = 0
    correct = 0
    with torch.no_grad(): # no_grad()를 통해 gradient 계산 안함 -> 메모리 및 속도 최적화
        for image, label in test_loader:
            image, label = image.to(DEVICE), label.to(DEVICE)
            output = model(image)
            test_loss += criterion(output, label).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(label).sum().item()

    test_loss /= len(test_loader.dataset) / BATCH_SIZE
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, accuracy

# 8. 학습 루프 실행
for epoch in range(1, EPOCHS + 1): # Epoch 수만큼 학습 및 평가 반복
    train(model, train_loader, optimizer, log_interval=200) #train()을 통해 로그 출력 
    test_loss, test_accuracy = evaluate(model, test_loader) #evaluate()를 통해 테스트 데이터셋에 대한 loss와 accuracy 계산
    print(f"\n[EPOCH {epoch}] Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%\n")

# 9. 최종 평가(최종 정확도 및 손실 출력)
final_test_loss, final_test_accuracy = evaluate(model, test_loader)

print("Final Test Loss: {:.4f}".format(final_test_loss))
print("Final Test Accuracy: {:.2f}%".format(final_test_accuracy))

# 10. confusion matrix 시각화
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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

class_names = [str(i) for i in range(10)]  # MNIST 클래스 이름: '0' ~ '9'
plot_confusion_matrix(model, test_loader, class_names)