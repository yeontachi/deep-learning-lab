import torch
import cv2
import numpy as np
from torchvision.datasets import CIFAR10
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image

# CIFAR-10 클래스 이름
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# 3단계 Transform 클래스 정의
class ObjectColor_BackgroundGrayTransform:
    def __call__(self, img):
        img_np = np.array(img)  # [H, W, C]
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # 윤곽선 검출 → 객체 마스크 생성
        edges = cv2.Canny(gray, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        mask = np.zeros_like(gray)
        cv2.drawContours(mask, contours, -1, 255, thickness=-1)
        mask_3ch = cv2.merge([mask] * 3)  # shape: [H, W, 3]

        # 객체 부분: 원본 이미지와 마스크 적용
        object_region = cv2.bitwise_and(img_np, mask_3ch)

        # 배경 부분: 흐릿한 흑백
        background_gray = cv2.GaussianBlur(gray, (7, 7), sigmaX=0)
        background_rgb = cv2.merge([background_gray] * 3)

        # 배경 마스크 (객체 영역 반전)
        inverted_mask = cv2.bitwise_not(mask_3ch)
        background_region = cv2.bitwise_and(background_rgb, inverted_mask)

        # 객체 + 배경 합성
        blended = cv2.add(object_region, background_region)

        # PIL → Tensor
        blended_pil = Image.fromarray(blended)
        return transforms.ToTensor()(blended_pil)

halfway_dataset = CIFAR10(root="./data", train=True, download=True, transform=ObjectColor_BackgroundGrayTransform())

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
    plt.suptitle("Object in Color + Background in Blurred Grayscale", fontsize=16)
    plt.show()

show_one_per_class_rgb(halfway_dataset, class_names)
