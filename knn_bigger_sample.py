import numpy as np
from tensorflow.keras.datasets import cifar10
from k_nn_utils import compute_distances_no_loops, predict_labels
import matplotlib.pyplot as plt

# ------------------------------------------
# 1. 데이터 전처리
# ------------------------------------------

(X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = cifar10.load_data()

# 데이터 샘플링 (train: 50000개, test: 500개)
num_training = 50000
num_test = 500

X_train = X_train_raw[:num_training]
y_train = y_train_raw[:num_training].flatten() # (50000,) 형태 <- 1차원 행렬로 평탄화.


X_test = X_test_raw[:num_test]
y_test = y_test_raw[:num_test].flatten() # (500,) 형태


X_train = np.reshape(X_train, (X_train.shape[0], -1)) 
X_test = np.reshape(X_test, (X_test.shape[0], -1)) 


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


print(f"훈련 데이터 형태: {X_train.shape}") # 예상: (50000, 3072)
print(f"테스트 데이터 형태: {X_test.shape}") # 예상: (500, 3072)


# ---------------------------------------------------
# 2. 거리 계산
# ---------------------------------------------------
print("거리 행렬 계산 중")
dists = compute_distances_no_loops(X_test, X_train) # 함수 호출


# --------------------------------------------------
# 3. 예측
# --------------------------------------------------
print("예측 및 voting 중")
y_test_pred = pred = predict_labels(dists, y_train, k=5)


# --------------------------------------------------
# 4. 정확도 측정
# --------------------------------------------------
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print(f"정확도: {accuracy * 100:.2f}%")

"""
정확도: 27.80%

Good: 무작위 추측보다 높은 정확도
CIFAR-10은 10개의 클래스를 가지고 있으므로, random하게 추측했을 때 정확도는 약 10%일 것임.
방금 만든 k-NN의 경우 무작위 추측의 약 2.8배의 정확도를 가지고 있음.

Bad: 정확도가 '아주 높지는 않음'. 
이유를 조사해보니 k-NN은 단순히 픽셀 값의 거리만을 비교하기 때문에 배경 색, 물체의 약간의 회전만 가해져도 다른 이미지라고 판단함.
따라서 태생적으로 고성능(예: 90% 이상-CNN)을 내기 어려움
"""


# ---------------------------------------------------------
# 5. k값별로 정확도를 계산: 가장 많이 사용되는 k값들이 후보
# ---------------------------------------------------------
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]
k_to_accuracies = {} # 계산 결과로 나온 정확도를 딕셔너리로 저장

print("k값별 정확도 계산을 시작함")

for k in k_choices:
    y_test_pred = predict_labels(dists, y_train, k=k)

    # 정확도 측정
    num_correct = np.sum(y_test_pred == y_test)
    accuracy = float(num_correct) / num_test
    k_to_accuracies[k] = accuracy
    print(f"k = {k:3d}, accuracy = {accuracy * 100:.2f}%")

"""
k =   1, accuracy = 27.40%
k =   3, accuracy = 27.20%
k =   5, accuracy = 27.80%
k =   8, accuracy = 27.40%
k =  10, accuracy = 28.20%
k =  12, accuracy = 25.60%
k =  15, accuracy = 27.20%
k =  20, accuracy = 27.40%
k =  50, accuracy = 25.20%
k = 100, accuracy = 22.20%
"""

# -------------------------------------------------
# 6. 결과를 그래프로 시각화
# -------------------------------------------------
plt.figure(figsize=(10,6)) # 가로 10'인치', 세로 6인치
plt.plot(k_to_accuracies.keys(), k_to_accuracies.values(), marker='o', linestyle='-', color='b')

# 가독성을 위한 그래프 꾸미기
plt.title('k-NN: Varying number of neighbors (K)')
plt.xlabel('Value of K')
plt.ylabel('Accuracy')
plt.xticks(k_choices) # 원하는 지점의 숫자를 정확히 보기 위해 눈금의 숫자를 k_choices로 지정
plt.grid(True) # 배경에 격자 생성

plt.show()