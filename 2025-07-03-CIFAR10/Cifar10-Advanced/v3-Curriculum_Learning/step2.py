import torch
import cv2
import numpy as np
from torchvision.datasets import CIFAR10
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image

# CIFAR-10 클래스 이름
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# CIFAR-10 정규화 값
mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)

# 단계: 객체 내부만 색 채우고 배경 제거
class FilledObjectTransform:
    def __call__(self, img):
        img_np = np.array(img)  # PIL → NumPy [H, W, C]
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # 윤곽선 검출
        edges = cv2.Canny(gray, 100, 200)

        # 외곽선 따라 Contour 추출
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 마스크 만들기 (배경 0, 객체는 흰색)
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, contours, -1, color=255, thickness=-1)  # 내부 채우기

        # 마스크를 3채널로 확장
        mask_rgb = cv2.merge([mask, mask, mask])  # shape: [H, W, 3]

        # 마스크 적용: 객체만 남기고 배경 제거
        result = cv2.bitwise_and(img_np, mask_rgb)

        # PIL 이미지 → Tensor
        result_pil = Image.fromarray(result)
        tensor = transforms.ToTensor()(result_pil)
        return tensor

# Transform 적용된 Dataset 생성
filled_dataset = CIFAR10(root="./data", train=True, download=True, transform=FilledObjectTransform())

# 클래스별로 하나씩 보여주는 함수
def show_one_per_class_rgb(dataset, class_names):
    class_seen = {i: False for i in range(10)}
    samples = [None for _ in range(10)]

    for image, label in dataset:
        if not class_seen[label]:
            samples[label] = (image, label)
            class_seen[label] = True
        if all(class_seen.values()):
            break

    plt.figure(figsize=(20, 4))
    for i, (image, label) in enumerate(samples):
        image = torch.clamp(image, 0, 1)
        plt.subplot(1, 10, i + 1)
        plt.imshow(image.permute(1, 2, 0).numpy())  # [C, H, W] → [H, W, C]
        plt.title(class_names[label])
        plt.axis('off')
    plt.suptitle("Filled Object Only CIFAR-10: One Sample per Class", fontsize=16)
    plt.show()

# 실행
show_one_per_class_rgb(filled_dataset, class_names)
