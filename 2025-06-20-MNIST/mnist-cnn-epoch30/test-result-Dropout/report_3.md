# MNIST CNN dropout 비교 분석(p=0.2~p=0.7)

## 실험 목적
본 실험의 목적은 CNN 기반 MNIST 분류 모델에 Dropout 기법을 적용함으로써 과적합 방지 효과와 일반화 성능 향상 여부를 분석하는 것이다. 일반적으로 Dropout은 학습 시 일부 뉴런을 확률적으로 제거함으로써 모델이 특정 뉴런에 과도하게 의존하는 것을 방지하여 일반화 성능을 높이는 데 사용된다. 특히 Dropout 확률이 너무 작거나 클 경우 학습 성능에 부정적인 영향을 줄 수 있기 때문에, 적절한 확률을 설정하는 것이 중요하다.

이에 따라 본 실험에서는 Dropout 확률을 0.2부터 0.7까지 다양하게 설정하고, 각각의 설정이 모델의 훈련 손실 감소 속도, 검증 정확도 변화, 그리고 **과적합 발생 시점(에폭 위치)**에 미치는 영향을 비교하였다. 일반적으로 Dropout 확률은 0.3~0.5 수준에서 최적 성능을 보이는 경향이 있다는 보고가 있어, 이를 기준으로 더 작거나 큰 확률을 적용했을 때 성능이 어떻게 변화하는지를 확인하고자 하였다.

## 실험 결과

### p=0.2
| Epoch | Train Loss | Train Acc (%) | Val Loss | Val Acc (%) |
| ----- | ---------- | ------------- | -------- | ----------- |
| 1     | 0.1871     | 94.15         | 0.0662   | 97.88       |
| 2     | 0.0632     | 98.06         | 0.0529   | 98.43       |
| 3     | 0.0462     | 98.58         | 0.0481   | 98.58       |
| 4     | 0.0350     | 98.88         | 0.0432   | 98.67       |
| 5     | 0.0282     | 99.12         | 0.0385   | 98.83       |
| 6     | 0.0243     | 99.19         | 0.0359   | 98.98       |
| 7     | 0.0200     | 99.35         | 0.0412   | 98.89       |
| 8     | 0.0175     | 99.41         | 0.0427   | 98.92       |
| 9     | 0.0139     | 99.53         | 0.0467   | 99.00       |
| 10    | 0.0141     | 99.52         | 0.0412   | 98.97       |
| 11    | 0.0119     | 99.59         | 0.0482   | 98.86       |
| 12    | 0.0114     | 99.64         | 0.0458   | 99.03       |
| 13    | 0.0113     | 99.64         | 0.0464   | 99.08       |
| 14    | 0.0090     | 99.66         | 0.0475   | 98.93       |
| 15    | 0.0093     | 99.66         | 0.0568   | 98.86       |
| 16    | 0.0075     | 99.74         | 0.0439   | 99.13       |
| 17    | 0.0088     | 99.70         | 0.0537   | 98.92       |
| 18    | 0.0074     | 99.74         | 0.0490   | 99.07       |
| 19    | 0.0062     | 99.78         | 0.0634   | 98.99       |
| 20    | 0.0060     | 99.80         | 0.0594   | 98.95       |
| 21    | 0.0068     | 99.77         | 0.0631   | 98.98       |
| 22    | 0.0062     | 99.80         | 0.0750   | 98.72       |
| 23    | 0.0058     | 99.80         | 0.0473   | 99.12       |
| 24    | 0.0049     | 99.81         | 0.0582   | 99.07       |
| 25    | 0.0055     | 99.83         | 0.0612   | 98.92       |
| 26    | 0.0047     | 99.86         | 0.0771   | 98.89       |
| 27    | 0.0058     | 99.82         | 0.0574   | 99.09       |
| 28    | 0.0057     | 99.83         | 0.0674   | 98.97       |
| 29    | 0.0052     | 99.86         | 0.0592   | 98.99       |
| 30    | 0.0046     | 99.84         | 0.0604   | 99.03       |

**Training Summary**
| Metric                 | Value  |
| ---------------------- | ------ |
| Average Train Loss     | 0.0198 |
| Average Train Accuracy | 99.36% |
| Average Val Loss       | 0.0532 |
| Average Val Accuracy   | 98.89% |
| Final Test Loss        | 0.0713 |
| Final Test Accuracy    | 98.50% |

### p=0.3
| Epoch | Train Loss | Train Acc (%) | Val Loss | Val Acc (%) |
| ----- | ---------- | ------------- | -------- | ----------- |
| 1     | 0.2001     | 93.65         | 0.0615   | 98.17       |
| 2     | 0.0699     | 97.91         | 0.0452   | 98.59       |
| 3     | 0.0503     | 98.46         | 0.0539   | 98.47       |
| 4     | 0.0385     | 98.81         | 0.0367   | 98.98       |
| 5     | 0.0301     | 99.03         | 0.0411   | 98.80       |
| 6     | 0.0255     | 99.19         | 0.0471   | 98.67       |
| 7     | 0.0224     | 99.25         | 0.0381   | 98.97       |
| 8     | 0.0201     | 99.34         | 0.0426   | 98.83       |
| 9     | 0.0156     | 99.46         | 0.0412   | 99.10       |
| 10    | 0.0152     | 99.45         | 0.0352   | 99.13       |
| 11    | 0.0121     | 99.59         | 0.0405   | 99.10       |
| 12    | 0.0112     | 99.60         | 0.0406   | 99.12       |
| 13    | 0.0102     | 99.64         | 0.0425   | 99.22       |
| 14    | 0.0094     | 99.69         | 0.0464   | 99.23       |
| 15    | 0.0089     | 99.68         | 0.0483   | 99.12       |
| 16    | 0.0075     | 99.73         | 0.0484   | 98.98       |
| 17    | 0.0092     | 99.68         | 0.0421   | 99.17       |
| 18    | 0.0068     | 99.78         | 0.0560   | 99.11       |
| 19    | 0.0068     | 99.75         | 0.0586   | 99.02       |
| 20    | 0.0070     | 99.77         | 0.0605   | 99.01       |
| 21    | 0.0068     | 99.77         | 0.0530   | 99.12       |
| 22    | 0.0072     | 99.76         | 0.0504   | 99.18       |
| 23    | 0.0063     | 99.77         | 0.0518   | 99.17       |
| 24    | 0.0059     | 99.81         | 0.0591   | 99.22       |
| 25    | 0.0058     | 99.80         | 0.0627   | 99.00       |
| 26    | 0.0054     | 99.83         | 0.0661   | 98.98       |
| 27    | 0.0044     | 99.83         | 0.0574   | 99.22       |
| 28    | 0.0069     | 99.80         | 0.0555   | 99.10       |
| 29    | 0.0050     | 99.85         | 0.0595   | 99.13       |
| 30    | 0.0050     | 99.85         | 0.0584   | 99.09       |

**Training Summary**
| Metric                     | Value  |
| -------------------------- | ------ |
| Average Train Loss         | 0.0212 |
| Average Train Accuracy (%) | 99.32% |
| Average Val Loss           | 0.0500 |
| Average Val Accuracy (%)   | 99.00% |
| Final Test Loss            | 0.0329 |
| Final Test Accuracy (%)    | 99.60% |

### p=0.4
| Epoch | Train Loss | Train Acc (%) | Val Loss | Val Acc (%) |
| ----- | ---------- | ------------- | -------- | ----------- |
| 1     | 0.2688     | 91.93         | 0.0675   | 97.84       |
| 2     | 0.0943     | 97.16         | 0.0528   | 98.33       |
| 3     | 0.0736     | 97.78         | 0.0410   | 98.72       |
| 4     | 0.0586     | 98.23         | 0.0393   | 98.76       |
| 5     | 0.0478     | 98.58         | 0.0347   | 98.96       |
| 6     | 0.0416     | 98.67         | 0.0323   | 99.01       |
| 7     | 0.0361     | 98.87         | 0.0359   | 99.04       |
| 8     | 0.0338     | 98.97         | 0.0364   | 98.99       |
| 9     | 0.0282     | 99.10         | 0.0500   | 98.80       |
| 10    | 0.0252     | 99.19         | 0.0474   | 98.81       |
| 11    | 0.0224     | 99.28         | 0.0374   | 99.08       |
| 12    | 0.0194     | 99.34         | 0.0416   | 99.01       |
| 13    | 0.0174     | 99.47         | 0.0485   | 98.88       |
| 14    | 0.0193     | 99.36         | 0.0403   | 99.08       |
| 15    | 0.0161     | 99.45         | 0.0397   | 99.08       |
| 16    | 0.0150     | 99.53         | 0.0504   | 99.01       |
| 17    | 0.0145     | 99.47         | 0.0443   | 99.00       |
| 18    | 0.0132     | 99.56         | 0.0553   | 99.01       |
| 19    | 0.0129     | 99.61         | 0.0433   | 99.17       |
| 20    | 0.0123     | 99.55         | 0.0452   | 99.07       |
| 21    | 0.0122     | 99.60         | 0.0444   | 99.17       |
| 22    | 0.0100     | 99.65         | 0.0513   | 99.07       |
| 23    | 0.0104     | 99.62         | 0.0462   | 99.14       |
| 24    | 0.0115     | 99.61         | 0.0458   | 99.05       |
| 25    | 0.0102     | 99.67         | 0.0534   | 99.02       |
| 26    | 0.0092     | 99.67         | 0.0595   | 99.05       |
| 27    | 0.0109     | 99.67         | 0.0451   | 99.17       |
| 28    | 0.0096     | 99.69         | 0.0578   | 99.12       |
| 29    | 0.0081     | 99.74         | 0.0634   | 98.97       |
| 30    | 0.0099     | 99.66         | 0.0471   | 99.17       |

**Training Summary**
| Metric                     | Value  |
| -------------------------- | ------ |
| Average Train Loss         | 0.0324 |
| Average Train Accuracy (%) | 98.99% |
| Average Val Loss           | 0.0466 |
| Average Val Accuracy (%)   | 98.95% |
| Final Test Loss            | 0.0493 |
| Final Test Accuracy (%)    | 99.20% |


### p=0.5
| Epoch | Train Loss | Train Acc (%) | Val Loss | Val Acc (%) |
| ----- | ---------- | ------------- | -------- | ----------- |
| 1     | 0.2857     | 91.11         | 0.0788   | 97.58       |
| 2     | 0.1069     | 96.90         | 0.0525   | 98.36       |
| 3     | 0.0775     | 97.67         | 0.0443   | 98.66       |
| 4     | 0.0643     | 98.05         | 0.0446   | 98.63       |
| 5     | 0.0504     | 98.49         | 0.0410   | 98.73       |
| 6     | 0.0448     | 98.65         | 0.0405   | 98.73       |
| 7     | 0.0402     | 98.80         | 0.0407   | 98.83       |
| 8     | 0.0354     | 98.93         | 0.0401   | 98.92       |
| 9     | 0.0294     | 99.08         | 0.0360   | 99.12       |
| 10    | 0.0272     | 99.14         | 0.0400   | 99.05       |
| 11    | 0.0249     | 99.19         | 0.0439   | 98.91       |
| 12    | 0.0234     | 99.23         | 0.0500   | 98.95       |
| 13    | 0.0222     | 99.30         | 0.0394   | 99.03       |
| 14    | 0.0191     | 99.35         | 0.0349   | 99.15       |
| 15    | 0.0176     | 99.43         | 0.0412   | 99.09       |
| 16    | 0.0167     | 99.43         | 0.0527   | 99.02       |
| 17    | 0.0162     | 99.45         | 0.0457   | 99.08       |
| 18    | 0.0157     | 99.49         | 0.0426   | 99.11       |
| 19    | 0.0154     | 99.50         | 0.0482   | 99.11       |
| 20    | 0.0129     | 99.54         | 0.0453   | 99.08       |
| 21    | 0.0147     | 99.51         | 0.0441   | 99.12       |
| 22    | 0.0118     | 99.60         | 0.0448   | 99.16       |
| 23    | 0.0117     | 99.62         | 0.0449   | 99.08       |
| 24    | 0.0122     | 99.58         | 0.0464   | 99.14       |
| 25    | 0.0110     | 99.61         | 0.0596   | 99.11       |
| 26    | 0.0096     | 99.67         | 0.0592   | 99.03       |
| 27    | 0.0117     | 99.64         | 0.0477   | 99.17       |
| 28    | 0.0105     | 99.69         | 0.0671   | 99.01       |
| 29    | 0.0109     | 99.66         | 0.0603   | 99.06       |
| 30    | 0.0110     | 99.66         | 0.0534   | 99.08       |

**Training Summary**
| Metric                 | Value  |
| ---------------------- | ------ |
| Average Train Loss     | 0.0354 |
| Average Train Accuracy | 98.90% |
| Average Val Loss       | 0.0477 |
| Average Val Accuracy   | 98.94% |
| Final Test Loss        | 0.0480 |
| Final Test Accuracy    | 99.10% |

### p=0.6
| Epoch | Train Loss | Train Acc (%) | Val Loss | Val Acc (%) |
|-------|------------|----------------|----------|--------------|
| 1     | 0.3093     | 90.12          | 0.0803   | 97.59        |
| 2     | 0.1281     | 96.15          | 0.0558   | 98.35        |
| 3     | 0.0969     | 97.07          | 0.0543   | 98.53        |
| 4     | 0.0803     | 97.66          | 0.0522   | 98.65        |
| 5     | 0.0680     | 97.95          | 0.0450   | 98.80        |
| 6     | 0.0617     | 98.15          | 0.0438   | 98.92        |
| 7     | 0.0500     | 98.46          | 0.0403   | 99.03        |
| 8     | 0.0470     | 98.47          | 0.0370   | 99.08        |
| 9     | 0.0402     | 98.73          | 0.0435   | 99.03        |
| 10    | 0.0384     | 98.74          | 0.0398   | 99.05        |
| 11    | 0.0355     | 98.86          | 0.0425   | 99.03        |
| 12    | 0.0319     | 98.96          | 0.0401   | 99.08        |
| 13    | 0.0327     | 98.97          | 0.0484   | 98.95        |
| 14    | 0.0308     | 99.05          | 0.0379   | 99.13        |
| 15    | 0.0273     | 99.12          | 0.0454   | 99.00        |
| 16    | 0.0228     | 99.23          | 0.0418   | 99.09        |
| 17    | 0.0246     | 99.21          | 0.0440   | 99.26        |
| 18    | 0.0240     | 99.22          | 0.0446   | 99.25        |
| 19    | 0.0232     | 99.22          | 0.0417   | 99.23        |
| 20    | 0.0201     | 99.34          | 0.0503   | 99.04        |
| 21    | 0.0199     | 99.35          | 0.0531   | 98.98        |
| 22    | 0.0201     | 99.36          | 0.0495   | 99.19        |
| 23    | 0.0185     | 99.36          | 0.0516   | 99.13        |
| 24    | 0.0200     | 99.33          | 0.0466   | 99.22        |
| 25    | 0.0162     | 99.45          | 0.0466   | 99.30        |
| 26    | 0.0173     | 99.42          | 0.0453   | 99.09        |
| 27    | 0.0148     | 99.49          | 0.0570   | 99.16        |
| 28    | 0.0164     | 99.47          | 0.0472   | 99.07        |
| 29    | 0.0147     | 99.51          | 0.0538   | 99.12        |
| 30    | 0.0159     | 99.53          | 0.0529   | 99.09        |

**Training Summary**
| Metric                | Value     |
|-----------------------|-----------|
| Average Train Loss    | 0.0456    |
| Average Train Accuracy| 98.56%    |
| Average Val Loss      | 0.0477    |
| Average Val Accuracy  | 98.98%    |
| Final Test Loss       | 0.0383    |
| Final Test Accuracy   | 99.10%    |

### p=0.7
| Epoch | Train Loss | Train Acc (%) | Val Loss | Val Acc (%) |
|-------|------------|----------------|----------|--------------|
| 1     | 0.4431     | 85.36          | 0.0754   | 97.81        |
| 2     | 0.2051     | 93.39          | 0.0635   | 98.23        |
| 3     | 0.1572     | 94.97          | 0.0543   | 98.39        |
| 4     | 0.1305     | 95.72          | 0.0513   | 98.61        |
| 5     | 0.1158     | 96.16          | 0.0470   | 98.65        |
| 6     | 0.1074     | 96.45          | 0.0438   | 98.82        |
| 7     | 0.0950     | 96.84          | 0.0453   | 98.83        |
| 8     | 0.0876     | 97.16          | 0.0525   | 98.77        |
| 9     | 0.0817     | 97.19          | 0.0518   | 98.88        |
| 10    | 0.0788     | 97.34          | 0.0429   | 98.80        |
| 11    | 0.0742     | 97.47          | 0.0399   | 98.89        |
| 12    | 0.0680     | 97.69          | 0.0476   | 99.02        |
| 13    | 0.0627     | 97.89          | 0.0455   | 98.99        |
| 14    | 0.0600     | 97.99          | 0.0456   | 98.94        |
| 15    | 0.0564     | 98.06          | 0.0452   | 99.03        |
| 16    | 0.0533     | 98.14          | 0.0486   | 98.97        |
| 17    | 0.0502     | 98.27          | 0.0560   | 98.88        |
| 18    | 0.0460     | 98.38          | 0.0472   | 99.07        |
| 19    | 0.0428     | 98.55          | 0.0494   | 99.09        |
| 20    | 0.0468     | 98.46          | 0.0483   | 99.06        |
| 21    | 0.0418     | 98.51          | 0.0480   | 99.06        |
| 22    | 0.0407     | 98.62          | 0.0513   | 99.05        |
| 23    | 0.0388     | 98.62          | 0.0555   | 98.96        |
| 24    | 0.0398     | 98.62          | 0.0537   | 99.08        |
| 25    | 0.0384     | 98.66          | 0.0493   | 99.05        |
| 26    | 0.0331     | 98.86          | 0.0582   | 99.00        |
| 27    | 0.0324     | 98.91          | 0.0537   | 99.05        |
| 28    | 0.0313     | 98.90          | 0.0557   | 99.01        |
| 29    | 0.0335     | 98.88          | 0.0539   | 99.08        |
| 30    | 0.0318     | 98.88          | 0.0525   | 99.05        |

**Training Summary**
| Metric                 | Value     |
|------------------------|-----------|
| Average Train Loss     | 0.0808    |
| Average Train Accuracy | 97.30%    |
| Average Val Loss       | 0.0511    |
| Average Val Accuracy   | 98.87%    |
| Final Test Loss        | 0.0241    |
| Final Test Accuracy    | 99.20%    |

### p==0.1
| Epoch | Train Loss | Train Acc (%) | Val Loss | Val Acc (%) |
| ----- | ---------- | ------------- | -------- | ----------- |
| 1     | 0.1794     | 94.44         | 0.0628   | 98.22       |
| 2     | 0.0564     | 98.28         | 0.0491   | 98.61       |
| 3     | 0.0402     | 98.73         | 0.0498   | 98.62       |
| 4     | 0.0291     | 99.07         | 0.0447   | 98.78       |
| 5     | 0.0239     | 99.22         | 0.0435   | 98.82       |
| 6     | 0.0174     | 99.48         | 0.0399   | 98.98       |
| 7     | 0.0154     | 99.47         | 0.0414   | 98.88       |
| 8     | 0.0136     | 99.52         | 0.0521   | 98.75       |
| 9     | 0.0107     | 99.66         | 0.0415   | 99.02       |
| 10    | 0.0101     | 99.67         | 0.0530   | 98.96       |
| 11    | 0.0085     | 99.71         | 0.0475   | 99.01       |
| 12    | 0.0068     | 99.78         | 0.0603   | 98.76       |
| 13    | 0.0082     | 99.73         | 0.0483   | 98.93       |
| 14    | 0.0060     | 99.80         | 0.0526   | 98.94       |
| 15    | 0.0061     | 99.80         | 0.0477   | 98.98       |
| 16    | 0.0055     | 99.84         | 0.0593   | 98.91       |
| 17    | 0.0077     | 99.75         | 0.0583   | 98.95       |
| 18    | 0.0043     | 99.84         | 0.0638   | 98.98       |
| 19    | 0.0037     | 99.88         | 0.0653   | 98.91       |
| 20    | 0.0044     | 99.87         | 0.0813   | 98.85       |
| 21    | 0.0048     | 99.84         | 0.0770   | 98.95       |
| 22    | 0.0051     | 99.83         | 0.0776   | 98.86       |
| 23    | 0.0032     | 99.90         | 0.0632   | 99.04       |
| 24    | 0.0053     | 99.84         | 0.0709   | 99.03       |
| 25    | 0.0035     | 99.89         | 0.0743   | 99.03       |
| 26    | 0.0045     | 99.86         | 0.0783   | 98.94       |
| 27    | 0.0038     | 99.88         | 0.0829   | 98.87       |
| 28    | 0.0023     | 99.94         | 0.0767   | 98.88       |
| 29    | 0.0038     | 99.89         | 0.0712   | 99.00       |
| 30    | 0.0039     | 99.89         | 0.0745   | 98.93       |

**Training Summary**
| Metric                     | Value  |
| -------------------------- | ------ |
| **Average Train Loss**     | 0.0166 |
| **Average Train Accuracy** | 99.48% |
| **Average Val Loss**       | 0.0603 |
| **Average Val Accuracy**   | 98.88% |
| **Final Test Loss**        | 0.0538 |
| **Final Test Accuracy**    | 98.60% |

### p==0.9
| Epoch | Train Loss | Train Acc (%) | Val Loss | Val Acc (%) |
|-------|------------|----------------|----------|--------------|
| 1     | 1.3297     | 45.93          | 0.2458   | 96.53        |
| 2     | 0.9626     | 60.53          | 0.1531   | 97.08        |
| 3     | 0.8287     | 66.47          | 0.1135   | 97.47        |
| 4     | 0.7611     | 69.39          | 0.0918   | 97.80        |
| 5     | 0.6924     | 72.30          | 0.0764   | 98.10        |
| 6     | 0.6342     | 75.17          | 0.0713   | 98.22        |
| 7     | 0.6124     | 75.95          | 0.0693   | 98.33        |
| 8     | 0.5812     | 77.44          | 0.0600   | 98.51        |
| 9     | 0.5575     | 78.40          | 0.0609   | 98.45        |
| 10    | 0.5454     | 79.09          | 0.0619   | 98.40        |
| 11    | 0.5305     | 79.69          | 0.0625   | 98.47        |
| 12    | 0.5282     | 79.60          | 0.0571   | 98.47        |
| 13    | 0.5201     | 80.05          | 0.0549   | 98.55        |
| 14    | 0.5144     | 80.35          | 0.0539   | 98.72        |
| 15    | 0.5082     | 80.57          | 0.0547   | 98.61        |
| 16    | 0.5035     | 81.26          | 0.0516   | 98.57        |
| 17    | 0.4946     | 81.18          | 0.0563   | 98.53        |
| 18    | 0.4880     | 81.67          | 0.0640   | 98.42        |
| 19    | 0.4772     | 82.03          | 0.0530   | 98.78        |
| 20    | 0.4813     | 81.94          | 0.0570   | 98.56        |
| 21    | 0.4783     | 82.16          | 0.0562   | 98.67        |
| 22    | 0.4789     | 82.34          | 0.0673   | 98.50        |
| 23    | 0.4725     | 82.43          | 0.0580   | 98.65        |
| 24    | 0.4607     | 82.97          | 0.0551   | 98.77        |
| 25    | 0.4591     | 82.99          | 0.0545   | 98.62        |
| 26    | 0.4590     | 82.94          | 0.0587   | 98.68        |
| 27    | 0.4552     | 83.06          | 0.0526   | 98.73        |
| 28    | 0.4497     | 83.57          | 0.0628   | 98.62        |
| 29    | 0.4410     | 83.59          | 0.0560   | 98.78        |
| 30    | 0.4394     | 83.89          | 0.0557   | 98.74        |

**Training Summary**
| Metric                 | Value     |
|------------------------|-----------|
| Average Train Loss     | 0.5715    |
| Average Train Accuracy | 77.96%    |
| Average Val Loss       | 0.0715    |
| Average Val Accuracy   | 98.38%    |
| Final Test Loss        | 0.0504    |
| Final Test Accuracy    | 99.10%    |

### p=1
| Epoch | Train Loss | Train Acc (%) | Val Loss | Val Acc (%) |
| ----- | ---------- | ------------- | -------- | ----------- |
| 1     | 2.3016     | 11.26         | 2.3002   | 11.14       |
| 2     | 2.3013     | 11.26         | 2.3005   | 11.15       |
| 3     | 2.3013     | 11.26         | 2.3005   | 11.15       |
| 4     | 2.3013     | 11.26         | 2.3007   | 11.15       |
| 5     | 2.3013     | 11.26         | 2.3005   | 11.15       |
| 6     | 2.3013     | 11.26         | 2.3003   | 11.15       |
| 7     | 2.3013     | 11.26         | 2.3002   | 11.15       |
| 8     | 2.3013     | 11.26         | 2.3003   | 11.15       |
| 9     | 2.3013     | 11.26         | 2.3001   | 11.15       |
| 10    | 2.3013     | 11.26         | 2.3001   | 11.15       |
| 11    | 2.3013     | 11.26         | 2.3003   | 11.15       |
| 12    | 2.3013     | 11.26         | 2.3001   | 11.15       |
| 13    | 2.3013     | 11.26         | 2.3002   | 11.15       |
| 14    | 2.3013     | 11.26         | 2.3001   | 11.07       |
| 15    | 2.3013     | 11.26         | 2.3004   | 11.15       |
| 16    | 2.3013     | 11.26         | 2.3002   | 11.15       |
| 17    | 2.3013     | 11.26         | 2.3003   | 11.15       |
| 18    | 2.3013     | 11.26         | 2.3002   | 11.15       |
| 19    | 2.3013     | 11.26         | 2.3002   | 11.15       |
| 20    | 2.3013     | 11.26         | 2.3001   | 11.15       |
| 21    | 2.3013     | 11.26         | 2.3002   | 11.15       |
| 22    | 2.3013     | 11.26         | 2.3002   | 11.15       |
| 23    | 2.3013     | 11.26         | 2.3004   | 11.15       |
| 24    | 2.3013     | 11.26         | 2.3003   | 11.15       |
| 25    | 2.3013     | 11.26         | 2.3003   | 11.15       |
| 26    | 2.3013     | 11.26         | 2.3006   | 11.15       |
| 27    | 2.3013     | 11.26         | 2.3004   | 11.15       |
| 28    | 2.3013     | 11.26         | 2.3004   | 11.15       |
| 29    | 2.3013     | 11.26         | 2.3002   | 11.15       |
| 30    | 2.3013     | 11.26         | 2.3003   | 11.15       |

**Trining Summary**
| Metric                     | Value  |
| -------------------------- | ------ |
| **Average Train Loss**     | 2.3013 |
| **Average Train Accuracy** | 11.26% |
| **Average Val Loss**       | 2.3003 |
| **Average Val Accuracy**   | 11.15% |
| **Final Test Loss**        | 2.3591 |
| **Final Test Accuracy**    | 10.00% |

## 실험 결과 및 분석 요약

![Alt text](/test-result/images/DropoutGraph_1.png)

![Alt text](/test-result/images/DropoutGraph_2.png)

![Alt text](/test-result/images/p1dropout.png)

 - **전반적인 경향성** : 드롭아웃 확률이 0.4~0.5일 때 가장 안정적인 성능을 보였으며, 학습 정확도와 검증 정확도 간의 차이가 적고 손실도 낮았다.
 - **Dropout 확률 증가에 따른 경향** 
    - **p=0.2~0.3** : 드롭아웃 확률이 너무 낮을 경우 학습 성능은 매우 좋지만 과적합이 빠르게 나타나는 경향이 있었다.
    - **p=0.4~0.6** : 적당한 확률에서는 학습 성능도 높고, 검증 정확도도 높아 가장 이상적인 형태를 보였다.
    - **p=0.7 이상** : 학습 정확도가 급격히 낮아지고 손실이 증가하면서 모델이 과도하게 드롭되어 학습 자체가 어려워짐을 확인했다. 그럼에도 불구하고 테스트 데이터셋(클래스당 100개) 기준으로는 p=0.7에서 가장 높은 정확도(99.20%)가 나오기도 했다. 이는 드롭아웃이 너무 강하게 걸린 경우 학습 데이터에는 부정적이지만, 아주 제한된 테스트 데이터에서는 의외의 일반화 효과가 발생했을 가능성이 있다.
    - **p=0.9** : 학습 정확도는 약 84% 수준에 그치고 손실도 매우 높았다. 검증 성능에서도 약간의 하락이 있었고, 테스트 성능도 오히려 p=0.7보다 떨어져 드롭아웃 확률을 과도하게 높이면 학습도, 일반화도 저해될 수 있다는 결론에 도달했다.

 - 결론적으로 적절한 드롭아웃 확률은 (p=0.4~0.6)이 가장 효과적이였으며, 너무 낮거나 너무 높으면 각각 과적합 또는 학습 저해라는 부작용이 발생할 수 있다.

 - 추가적으로 드롭아웃 확률이 1일 때 모든 뉴런을 드롭(=끄는) 의미이므로 학습이 불가능할 것이라고 생각함. 실제로는 코드는 멈추지 않고 실행되고, 결과도 출력됨. > Pytorch의 Dropout는 학습 중에도 완전히 꺼버리진 않도록 처리. 즉, nn.Dropout(p=0.1)은 확률적으로 모든 뉴런을 끄도록 정의되지만, 실제로는 내부 구현에서 "완전히 모든 뉴런을 0으로 만들지 않도록하는 방어 로직"이 있다. 내부적으로 random mask가 전부 0이 되더라도, gradient 흐름이 완전히 막히는걸 방지하도록 되어 있다. 하지만 여전히 출력은 거의 전부 0에 가까우므로, 학습 자체는 거의 정보 전달이 안 되어 무의미해진다.
    - 실행되는 이유:
        - Pytorch 내부 로직(드롭아웃이 모두 0이 되더라도 에러 안 나게 처리)
        - 평가 시 드롭아웃 꺼짐(model.eval()에서는 전체 뉴런 사용)
        - 사용성 보장(극단적 설정에도 코드가 멈추지 않도록 설계)