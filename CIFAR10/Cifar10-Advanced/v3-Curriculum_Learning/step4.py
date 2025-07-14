import torch
import numpy as np
from torchvision.datasets import CIFAR10
from torchvision import transforms
import matplotlib.pyplot as plt

# CIFAR-10 클래스 이름
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# CIFAR-10 정규화 값
mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)

# ✅ 3단계: 원본 이미지 그대로 (정규화 포함)
original_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# 데이터셋 생성
original_dataset = CIFAR10(root="./data", train=True, download=True, transform=original_transform)

# ✅ 시각화를 위해 정규화 해제하는 함수 포함
def show_one_per_class_original(dataset, class_names):
    class_seen = {i: False for i in range(10)}
    samples = [None for _ in range(10)]

    for image, label in dataset:
        if not class_seen[label]:
            samples[label] = (image, label)
            class_seen[label] = True
        if all(class_seen.values()):
            break

    # 정규화 해제
    unnormalize = transforms.Normalize(
        mean=[-m/s for m, s in zip(mean, std)],
        std=[1/s for s in std]
    )

    plt.figure(figsize=(20, 4))
    for i, (image, label) in enumerate(samples):
        image = unnormalize(image)
        image = torch.clamp(image, 0, 1)

        plt.subplot(1, 10, i + 1)
        plt.imshow(image.permute(1, 2, 0).numpy())
        plt.title(class_names[label])
        plt.axis('off')
    plt.suptitle("Original CIFAR-10 Image (Full object and background)", fontsize=16)
    plt.show()

show_one_per_class_original(original_dataset, class_names)
