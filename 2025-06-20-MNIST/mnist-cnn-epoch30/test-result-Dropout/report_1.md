# MNIST CNN EPOCH(100 -> 30) 실험 보고서

## 실험 개요
 - **모델 구조** : 2층 CNN(Conv -> ReLU -> MaxPool) + 2층 완전 연결층(FC)
 - **데이터셋** : MNIST(28x28 크기의 흑백 손글씨 숫자 이미지, 총 10개 클래스)
 - **프레임워크** : PyTorch
 - **학습 Epoch 수** : 30
 - **Optimizer** : Adam(Learning rate = 0.001)
 - **활성화 함수** : ReLU
 - **정규화 기법** : 없음
 - **손실 함수** : CrossEntropyLoss(다중 클래스 분류에서 일반적으로 사용)
 - **평가 지표** : CrossEntropyLoss, Accuracy, Confusion Matrix

 ## 에폭에 따른 loss와 acc 비교 표

| Epoch | Train Loss | Train Acc (%) | Val Loss | Val Acc (%) |
|-------|------------|----------------|----------|--------------|
| 1     | 0.1565     | 95.16          | 0.0679   | 97.87        |
| 2     | 0.0482     | 98.47          | 0.0548   | 98.28        |
| 3     | 0.0335     | 98.98          | 0.0480   | 98.57        |
| 4     | 0.0229     | 99.25          | 0.0409   | 98.88        |
| 5     | 0.0176     | 99.42          | 0.0496   | 98.63        |
| 6     | 0.0133     | 99.56          | 0.0493   | 98.77        |
| 7     | 0.0114     | 99.63          | 0.0458   | 98.88        |
| 8     | 0.0092     | 99.71          | 0.0469   | 98.89        |
| 9     | 0.0073     | 99.77          | 0.0486   | 99.00        |
| 10    | 0.0068     | 99.77          | 0.0541   | 98.81        |
| 11    | 0.0061     | 99.80          | 0.0543   | 98.88        |
| 12    | 0.0058     | 99.81          | 0.0747   | 98.56        |
| 13    | 0.0041     | 99.87          | 0.0637   | 98.78        |
| 14    | 0.0038     | 99.87          | 0.0613   | 99.02        |
| 15    | 0.0035     | 99.90          | 0.0727   | 98.73        |
| 16    | 0.0041     | 99.87          | 0.0675   | 98.92        |
| 17    | 0.0042     | 99.87          | 0.0685   | 98.98        |
| 18    | 0.0036     | 99.90          | 0.0711   | 98.78        |
| 19    | 0.0034     | 99.88          | 0.0828   | 98.81        |
| 20    | 0.0027     | 99.92          | 0.0668   | 98.83        |
| 21    | 0.0040     | 99.89          | 0.0910   | 98.72        |
| 22    | 0.0021     | 99.93          | 0.0750   | 98.92        |
| 23    | 0.0025     | 99.93          | 0.0740   | 98.94        |
| 24    | 0.0035     | 99.88          | 0.0804   | 98.91        |
| 25    | 0.0022     | 99.93          | 0.0818   | 98.92        |
| 26    | 0.0014     | 99.95          | 0.0945   | 98.82        |
| 27    | 0.0029     | 99.90          | 0.0825   | 98.85        |
| 28    | 0.0035     | 99.91          | 0.0864   | 98.95        |
| 29    | 0.0030     | 99.91          | 0.0874   | 98.83        |
| 30    | 0.0019     | 99.95          | 0.0913   | 98.91        |

최종 체스트 정확도: 약 99.0% 내외

## 성능 분석

 - **훈련 성능** : 에폭이 증가함에 따라 Train Loss는 지속적으로 감소, 정확도는 거의 100%에 도달함. 특히 Epoch 9-10이후부터는 Train Loss가 매우 낮고 안정적임

 - **검증 성능** : Epoch 8 까지는 Val Loss가 안정적으로 하락함. 그러나 Epoch10부터 미세한 상능, Epoch12 이후에는 Val Loss가 눈에 띄게 증가하는 추세를 확인. Val Acc는 대체로 99.8~99.0% 사이에서 정체, Epoch이 늘어도 큰 개선이 없음

## 과적합 여부 분석
 - **과적합 발생 시점** : 약 Epoch 12~13 이후, 이후로는 Train Loss는 계속 낮아지지만, Val Loss는 상승세. Val Acc는 정체 또는 소폭 하락
 - **전형적인 과적합 패턴** : 모델이 훈련데이터를 지나치게 외우기 시작. 새로운 데이터(검증 데이터셋)에 대한 일반화 성능은 더 이상 개선되지 않는다.

## 결론 및 개선 방향
30 에폭 기준 CNN 모델은 훈련에는 매우 뛰어난 성능을 보였으나, 일반화 성능은 에폭 12~13 이후 정체되는 모습을 보인다.

과적합 방지를 위해 다음과 같은 개선이 필요
 - EarlyStopping 도입(예: patience=5)
 - 정규화 기버버 사용(예: Dropout)
 - 데이터 증강 활용
 - 더 복잡한 모델보다는 적절한 capacity 유지

제안된 EarlyStopping 시점
 - Val Loss의 증가가 3-5 에폭 연속 지속되는 시점
 - 실제로는 Epoch 13-14 전후에서 조기 종료하는 것이 적절한 전략으로 보임