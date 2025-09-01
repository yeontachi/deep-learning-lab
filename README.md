# 🧠 Deep Learning Lab

This repository serves as a personal laboratory for deep learning experimentation, research practice, and hands-on model implementation. It includes a variety of neural network experiments using **PyTorch** and other open-source tools, focusing on understanding model behavior through architectural variations, dataset manipulations, and training strategies.

---

## 🧪 Structure & Navigation

Each experiment is stored in a **timestamped directory** for easy chronological tracking.  
The directory naming convention is:

### 🗂️ Examples:
- `2025-06-30-MNIST` → Experiment using the MNIST dataset
- `2025-07-03-DataAugment` → Experiment with data augmentation techniques
- `2025-07-08-MobileNetV2` → Transfer learning with MobileNetV2
- `2025-07-10-BilliardDetection` → Vision-based object detection practice

> ✅ Each folder includes all relevant code, results, and documentation for the corresponding experiment.

---

## 📌 What’s Included

- 📚 Model architecture comparisons (MLP, CNN, etc.)
- ⚙️ Hyperparameter tuning and optimizer tests
- 🧪 Effects of activation functions and regularization (Dropout, BatchNorm)
- 🔄 Dataset transformations (noise, rotation, scaling)
- 📊 Result visualization (accuracy plots, confusion matrix, prediction samples)
- 📁 Open-source project analysis and customization

---

## 🚀 Frameworks & Tools

- [PyTorch](https://pytorch.org/)
- [Torchvision](https://pytorch.org/vision/)
- [Matplotlib / Seaborn](https://matplotlib.org/)
- [scikit-learn](https://scikit-learn.org/)
- Possibly [OpenCV](https://opencv.org/) or other tools in future tasks

---

## 🎯 Purpose

This lab is designed to:
- Deepen understanding of deep learning theory and practice
- Analyze how different design choices affect model performance
- Prepare for advanced research or academic projects in AI and computer vision

---

## Quantization Roadmap

| 시기                          | 주요 모델                                | 대표 연구/논문                                                                                                                                                                                                               | 핵심 아이디어                                   | 학습/실습 포인트                                                                  |
| --------------------------- | ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- | -------------------------------------------------------------------------- |
| **2019–2020** (CNN 중심)      | MobileNet, ResNet, EfficientNet      | - **HAWQ (2019)**: 레이어별 민감도 기반 mixed-precision<br>- **HAQ (2019)**: RL 기반 HW-aware quantization<br>- **LSQ (2019)**: step size 학습 QAT<br>- **DFQ/ZeroQ (2019–2020)**: 데이터 없는 PTQ<br>- **AdaRound (2020)**: 적응형 라운딩 PTQ | CNN을 INT8로 안정화<br>무데이터·혼합정밀 기법 등장         | PyTorch `torch.quantization` 활용<br>ResNet/MobileNet FP32 vs INT8 정확도·속도 비교 |
| **2021–2022** (ViT 진입)      | Vision Transformer (ViT, DeiT, Swin) | - **BRECQ (2021)**: 블록 단위 재구성 PTQ<br>- **PTQ4ViT (2022)**: ViT 8-bit 안정화<br>- **Q-ViT (2022)**: QAT 기반 저비트 ViT                                                                                                         | CNN 대비 분포 특성이 다른 ViT에 맞춘 PTQ/QAT 기법       | HuggingFace ViT 모델 불러와 PTQ 적용<br>FP32 vs INT8 정확도 비교                       |
| **2023** (LLM과의 접점, 초저비트)   | CLIP, ViT+Text                       | - **AWQ (2023)**: 활성 기반 weight-only 4bit<br>- **QLoRA (2023)**: 4bit + LoRA (LLM 영향, VLM에도 적용)                                                                                                                         | 비전-언어 모델에도 4bit 적용 가능성 열림                 | OpenAI CLIP 모델에 4bit quantization 적용 실습                                    |
| **2024** (구조적·멀티모달 확장)      | MobileNetV4, SmolVLA                 | - **Structured Quantization (2024)**: 채널·행렬 단위 양자화<br>- **MobileNetV4 (2024)**: 양자화 친화 구조 설계<br>- **SmolVLA (2024)**: 소형 VLA 모델                                                                                        | 단순 비트 축소를 넘어 구조 최적화<br>멀티모달 모델 엣지 디바이스 적용 | ONNX/TensorRT 기반 INT8 최적화<br>모바일/IoT 환경 배포 실습                              |
| **2025** (초저비트 안정화 + HW 결합) | MicroViT, OpenVLA int4, RepNeXt      | - **APhQ-ViT (CVPR 2025)**: ViT PTQ 성능 개선<br>- **OpenVLA int4 (2025)**: 4bit 멀티모달<br>- **RepNeXt (2025)**: HW-aware quantization 적용                                                                                    | 2bit 이하 초저비트 안정화, HW와의 긴밀한 결합             | PyTorch+ONNX로 ViT 4bit PTQ/QAT 실습<br>NPU/EdgeTPU에서 추론 성능 확인                |

---
## 🧑‍💻 Author

> JaeHyuck Yeon (연재혁)  
> Undergraduate Researcher — Deep Learning & AI Practice  
> Contact: [yeonjjhh@gmail.com](mailto:yeonjjhh@gmail.com)

---

Feel free to clone the repo, browse through the dated experiment folders, and learn from the process!
