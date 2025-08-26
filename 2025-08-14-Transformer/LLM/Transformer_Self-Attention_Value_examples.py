import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

# 예시 단어 벡터 (Word2Vec 가정)
# 실제로는 수백~수천 차원이지만, 예시는 5차원짜리 가짜 벡터 생성
# QK V 연산 결과 예시 
word_vectors = {
    "I": np.array([1.0, 0.5, 0.2, 0.1, 0.0]),
    "go": np.array([0.0, 1.0, 0.8, 0.2, 0.1]),
    "home": np.array([0.9, 0.7, 0.3, 0.5, 0.2]),
    "school": np.array([0.8, 0.6, 0.4, 0.7, 0.3]),
    "eat": np.array([0.1, 0.9, 0.7, 0.2, 0.3]),
    "food": np.array([0.7, 0.8, 0.5, 0.6, 0.4]),
    "cat": np.array([0.6, 0.3, 0.4, 0.2, 0.8]),
    "dog": np.array([0.5, 0.2, 0.5, 0.3, 0.9])
}

# 단어와 벡터 분리
words = list(word_vectors.keys())
vectors = np.array(list(word_vectors.values()))

# PCA로 2차원 축소
pca = PCA(n_components=2)
vectors_2d = pca.fit_transform(vectors)

# 시각화
plt.figure(figsize=(8, 6))
plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c='blue')

# 단어 라벨 붙이기
for i, word in enumerate(words):
    plt.text(vectors_2d[i, 0] + 0.02, vectors_2d[i, 1] + 0.02, word, fontsize=12)

plt.title("Word Embedding Space (PCA 2D projection)", fontsize=14)
plt.xlabel("PC1 (의미 축 1)")
plt.ylabel("PC2 (의미 축 2)")
plt.grid(True)
plt.show()