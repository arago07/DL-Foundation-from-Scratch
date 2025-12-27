import numpy as np
from tensorflow.keras.datasets import cifar10

# -------------------------------------
# 1. 데이터 전처리 및 점수 계산
# -------------------------------------

# 데이터 로드 및 전처리
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 이미지 펼치기
X = x_train.reshape(x_train.shape[0], -1).astype('float32')

N = X.shape[0] # 50,000
D = X.shape[1] # 3,072
C = 10

W = np.random.randn(D, C) * 0.01 # 0.01 곱한 것 <- overflow or loss 폭발 방지
b = np.random.randn(1, C)

# 점수 계산
scores = X.dot(W) + b # (50000,10)

# ----------------------------------------
# 2. 정답 점수 골라내기
# ----------------------------------------

y = y_train.flatten() 
correct_class_scores = scores[np.arange(N),y] # (모델의 답, 실제 답) 순서쌍을 원소로 가지는 새로운 행렬 생성

# 모든 클래스와의 점수 차이 계산
margins = scores - correct_class_scores.reshape(N, 1) + 1 # +1을 하는 이유: 정답 점수가 '확실하게' 높아야만 인정하기 위해 사용하는 관습적인 값
margins = np.maximum(0, margins)

# 정답 클래스의 경우 항상 loss가 1이 발생하므로 이를 0으로 초기화
margins[np.arange(N), y] = 0

# 최종 loss 구하기
loss = np.sum(margins) / N

# --------------------------------------
# 3. 결과 출력
# --------------------------------------

print(f"데이터 개수(N): {N}개")
print(f"가중치 크기(W): {W.shape}")
print(f"최종 Loss: {loss}")

# --------------------------------------
# 4. 결과 분석
# --------------------------------------
# 데이터 개수(N): 50000개
# 가중치 크기(W): (3072, 10)

# 최종 Loss:
# 실험 1: 338.4047027096444
# 실험 2: 305.9345332303497
# 실험 3: 312.17930413637566

# Loss가 지나치게 크다. 이후 optimization 할 예정
