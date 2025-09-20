# main_qat.py
import torch
import torch.nn as nn
from torchvision.models import resnet18
import copy

# 이전 예제와 동일한 헬퍼 함수들 (get_4bit_qconfig, adapt_resnet18_for_cifar10, ...)
# ... (이전 코드의 헬퍼 함수들을 여기에 복사) ...

# 헬퍼 함수 복사 영역 시작 
# --------------------------------------------------------------------------------------------------
from torch.quantization import FakeQuantize, QConfig, prepare_qat, convert
import torchvision
import torchvision.transforms as transforms
from quant.quant_layer import QuantLinear # *** 커스텀 레이어 임포트 ***

class FourBitFakeQuantize(FakeQuantize):
    def __init__(self, observer, **observer_kwargs):
        super().__init__(observer=observer, quant_min=0, quant_max=15, **observer_kwargs)

def get_4bit_qconfig():
    observer = torch.quantization.MinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_symmetric)
    return QConfig(activation=FourBitFakeQuantize.with_args(observer=observer), weight=FourBitFakeQuantize.with_args(observer=observer))

def adapt_resnet18_for_cifar10(model):
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    return model

def evaluate_model(model, data_loader, device):
    model.to(device)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=1):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0: print(f"E[{epoch+1}], S[{i+1}/{len(train_loader)}]")
    return model
# --------------------------------------------------------------------------------------------------
# 헬퍼 함수 복사 영역 끝


# *** 핵심 변환 함수 ***
def convert_to_quant_model(qat_model, device):
    qat_model.eval().to('cpu')
    # 1. FakeQuantize 모듈을 제거하고 가중치/활성화 통계만 남김
    converted_model = convert(qat_model, inplace=False)

    # 2. nn.Linear를 QuantLinear로 교체하고 가중치 패킹
    quant_model = adapt_resnet18_for_cifar10(resnet18()) # 새 뼈대
    
    # Conv 레이어 가중치는 그대로 복사 (커널 구현 안했으므로)
    quant_model.load_state_dict(converted_model.state_dict(), strict=False)

    # FC 레이어를 QuantLinear로 교체 및 가중치 패킹
    fc_fp_weight = converted_model.fc.weight()
    fc_scale = converted_model.fc.scale
    
    # 양자화 및 패킹
    fc_int_weight = torch.clamp(torch.round(fc_fp_weight / fc_scale.view(-1, 1)), 0, 15).to(torch.int32)
    
    packed_weight = torch.zeros(fc_int_weight.shape[0], fc_int_weight.shape[1] // 8, dtype=torch.int32)
    for i in range(8):
        packed_weight |= (fc_int_weight[:, i::8] << (i * 4))

    # QuantLinear 레이어 생성 및 가중치/스케일 할당
    quant_fc = QuantLinear(converted_model.fc.in_features, converted_model.fc.out_features)
    quant_fc.weight = packed_weight
    quant_fc.scale = fc_scale.view(-1)
    
    quant_model.fc = quant_fc
    return quant_model.to(device)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 데이터셋 로드 (이전과 동일)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    # STEP 1: FP32 모델 로드 및 평가
    print("\n[STEP 1] FP32 ResNet18 Baseline")
    fp32_model = adapt_resnet18_for_cifar10(resnet18(weights='IMAGENET1K_V1'))
    fp32_accuracy = evaluate_model(fp32_model, test_loader, device)
    print(f"  > FP32 Accuracy: {fp32_accuracy:.2f}%")

    # STEP 2 & 3: QAT 준비 및 미세조정
    print("\n[STEP 2 & 3] QAT Fine-tuning")
    qat_model = copy.deepcopy(fp32_model)
    qat_model.train()
    qat_model.qconfig = get_4bit_qconfig()
    prepare_qat(qat_model, inplace=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(qat_model.parameters(), lr=0.001, momentum=0.9)
    qat_model = train_model(qat_model, train_loader, criterion, optimizer, device, num_epochs=1)

    # STEP 4: 커스텀 QuantLinear 레이어를 사용하는 모델로 변환
    print("\n[STEP 4] Converting to model with custom QuantLinear layer")
    quant_model = convert_to_quant_model(qat_model, device)

    # STEP 5: 최종 성능 평가 (커스텀 커널 사용)
    print("\n[STEP 5] Evaluating final model using custom CUDA kernel")
    quant_accuracy = evaluate_model(quant_model, test_loader, device)
    
    print("\n" + "="*50)
    print("                     RESULTS")
    print("="*50)
    print(f"  > FP32 Baseline Model Accuracy: {fp32_accuracy:.2f}%")
    print(f"  > 4-bit QAT Model Accuracy (w/ CUDA Kernel): {quant_accuracy:.2f}%")
    print("="*50)
