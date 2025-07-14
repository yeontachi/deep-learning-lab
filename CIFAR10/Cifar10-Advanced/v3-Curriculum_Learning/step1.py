import cv2
import numpy as np
from torchvision.datasets import CIFAR10
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image

# Canny Edge 변환을 위한 Custom Transform
class EdgeTransform:
    def __init__(self, resize=(32, 32)):
        self.resize = resize

    def __call__(self, img):
        # PIL 이미지를 NumPy array로 변환
        img = np.array(img)

        # RGB → Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Canny edge detection
        edges = cv2.Canny(gray, threshold1=100, threshold2=200)

        # 1채널 이미지로 맞추고 텐서로 변환
        edges = Image.fromarray(edges)  # PIL 이미지로 다시 변환
        edges = edges.convert("L")  # 1채널
        tensor = transforms.ToTensor()(edges)  # [1, 32, 32] 형태
        return tensor

# EdgeTransform 적용된 CIFAR10 Dataset
edge_dataset = CIFAR10(root="./data", train=True, download=True, transform=EdgeTransform())

# 일부 샘플을 시각화해서 확인
def show_one_per_class_edge(dataset, class_names):
    # 각 클래스를 하나씩 담을 딕셔너리
    class_seen = {i: False for i in range(10)}
    samples = [None for _ in range(10)]

    for image, label in dataset:
        if not class_seen[label]:
            samples[label] = (image, label)
            class_seen[label] = True
        if all(class_seen.values()):
            break

    # 시각화
    plt.figure(figsize=(20, 4))
    for i, (image, label) in enumerate(samples):
        plt.subplot(1, 10, i + 1)
        plt.imshow(image.squeeze().numpy(), cmap='gray')
        plt.title(class_names[label])
        plt.axis('off')
    plt.suptitle("EdgeTransformed CIFAR-10: One Sample per Class", fontsize=16)
    plt.show()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

show_one_per_class_edge(edge_dataset, class_names)
