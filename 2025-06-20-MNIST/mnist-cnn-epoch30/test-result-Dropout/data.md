batch size 

| batch size | train  |   test |
| :--------- | :----: | -----: |
| 1          | 98.88% | 98.27% |
| 8          | 99.36% | 98.40% |
| 32         | 99.88% | 99.15% |
| 64         | 99.88% | 99.17% |
| 128        | 99.73% | 99.16% |
| 256        | 99.79% | 99.17% |
| 512        | 99.62% | 99.08% |
| 1024       | 99.22% | 98.95% |

activation func
| activation func | train  |   test |
| :-------------- | :----: | -----: |
| sigmoid         | 99.49% | 98.74% |
| ReLU            | 99.76% | 99.17% |
| tanh            | 99.88% | 99.00% |
| Leaky ReLU      | 99.74% | 99.15% |


| batch size | train  |   test |
| :--------- | :----: | -----: |
| 1          | 98.88% | 98.27% |
| 8          | 99.36% | 98.40% |
| 32         | 99.88% | 99.15% |
| 64         | 99.88% | 99.17% |
| 128        | 99.73% | 99.16% |
| 256        | 99.79% | 99.17% |
| 512        | 99.62% | 99.08% |
| 1024       | 99.22% | 98.95% |
| 2048       | 98.40% | 98.21% |
| 4096       | 97.64% | 97.75% |

with schedular
| Learning Rate | 평균 Loss | 평균 Accuracy (%) |
|---------------|------------|--------------------|
| 0.0010        | 0.03132    | 98.987             |
| 0.0025        | 0.03075    | 99.148             |
| 0.0050        | 0.03757    | 98.960             |

| Learning Rate | 평균 Loss | 평균 Accuracy (%) |
|---------------|------------|--------------------|
| 0.0010        | 0.03556    | 98.974             |
| 0.0025        | 0.04085    | 98.871             |
| 0.0050        | 0.05922    | 98.534             |


| Gamma | 평균 Loss | 평균 Accuracy (%) |
|-------|------------|--------------------|
| 0.600 | 0.09034    | 97.227             |
| 0.700 | 0.05332    | 98.588             |
| 0.725 | 0.06653    | 98.083             |
| 0.750 | 0.06754    | 97.990             |
| 0.800 | 0.10716    | 96.742             |


| batch size | test Acc |
| :--------- | -------: |
| 1          |   98.27% |
| 2          |   98.77% |
| 4          |   99.20% |
| 8          |   99.24% |
| 16         |   99.29% |
| 32         |   99.38% |
| 64         |   99.29% |
| 128        |   99.40% |
| 256        |   99.26% |
| 512        |   99.16% |
| 1024       |   99.12% |
| 2048       |   99.19% |
| 4096       |   98.82% |