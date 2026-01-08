import numpy as np
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import os

# -------------------------------------
# 1. 데이터 전처리 및 점수 계산
# -------------------------------------

# 데이터 로드 및 전처리
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 이미지 펼치기
X = x_train.reshape(x_train.shape[0], -1).astype('float32')
X /= 255.0 # 6. optimization 작성 후 데이터 스케일링을 위해 255로 나누는 식 추가

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

# ----------------------------------------
# 5. Optimization: gradient 계산 로직 추가
# -----------------------------------------

# loss > 0을 1로 표시하는 binary 행렬 생성
binary = margins
binary[margins > 0] = 1 # (N, 10) 형태

# 각 행마다 loss > 0 인 오답 클래스의 개수 카운트
row_sum = np.sum(binary, axis=1) # (N,) 형태가 되도록 더함
# y(정답 위치)에서 각각 해당 개수만큼 빼기
binary[np.arange(N), y] = - row_sum

# 기울기(dW) 계산
dW = X.T.dot(binary) # (N, D) -> (D, N) 형태로 transpose(전치)
dW /= N # 평균 기울기

# 결과 출력
print(f"기울기 dW의 크기: {dW.shape}")
print(f"dW의 평균 절댓값: {np.mean(np.abs(dW))}") # 업데이트 규모(magnitude) 확인
print(f"dW의 첫 5개 값: {dW.flatten()[:5]}")

# 기울기 dW의 크기: (3072, 10)
# dW의 평균 절댓값: 54.830119674479164
# dW의 첫 5개 값: [  -1.04418 -104.66138   14.81854  -38.50052  115.80954]

# ------------------------------------
# 6. Optimization: 학습 루프 추가
# ------------------------------------

# learning_rate = 1e-5 # 0.00001 <- 속도를 위해 증가
learning_rate = 1e-3
reg = 0.5 # 규제 강도
"""
print("\n학습 시작...")
for i in range(101): # 100번 반족
    # 점수 계산
    scores = X.dot(W) + b
    
    # loss, margin 계산
    correct_class_scores = scores[np.arange(N), y].reshape(N,1)
    margins = np.maximum(0, scores - correct_class_scores + 1)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N

    # gradient 계산
    binary = (margins > 0).astype(float)
    row_sum = np.sum(binary, axis=1)
    binary[np.arange(N), y] = - row_sum
    dW = X.T.dot(binary) / N
    db = np.sum(binary, axis=0) # b에 대한 미분 추가

    # parameter update
    W -= learning_rate * dW
    b -= learning_rate * db

    # 가독성과 성능을 위해 출력은 10번마다
    if i % 10 == 0:
        print(f"Iteration {i}: Loss는 {loss:.4f}")
"""

"""
결과 출력
데이터 개수(N): 50000개
가중치 크기(W): (3072, 10)
최종 Loss: 10.63846182700652
기울기 dW의 크기: (3072, 10)
dW의 평균 절댓값: 0.22097077582539196
dW의 첫 5개 값: [-0.35558189  0.17308126 -0.2649673   0.20014793  0.26133993]

학습 시작...
Iteration 0: Loss는 10.6385
Iteration 10: Loss는 8.9452
Iteration 20: Loss는 8.9310
Iteration 30: Loss는 8.9177
Iteration 40: Loss는 8.9045
Iteration 50: Loss는 8.8913
Iteration 60: Loss는 8.8782
Iteration 70: Loss는 8.8652
Iteration 80: Loss는 8.8521
Iteration 90: Loss는 8.8390
Iteration 100: Loss는 8.8259

Loss가 점점 감소했다! (10.6385 -> 8.8259)
"""

# -------------------------------------
# 7. Optimization
# -------------------------------------
batch_size = 200
num_iters = 1500 # 반복 횟수
# 검증 데이터 분리
num_validation = 1000
X_val = X[:num_validation]
y_val = y[:num_validation]
X_train_subset = X[num_validation:]
y_train_subset = y[num_validation:]

# 후보군 설정
learning_rates = [1e-3, 5e-4]
reg_strengths = [0.1, 0.25, 0.5]

results = {}
best_val = -1
best_W, best_b = None, None

print("\n하이퍼파라미터 튜닝 시작")

def predict(X, W, b):
    # 학습된 가중치를 사용해 입력 데이터의 클래스 예측

    # 점수 계산(N, C)
    scores = X.dot(W) + b
    # 가장 높은 점수를 가진 인덱스 선택
    y_pred = np.argmax(scores, axis=1) # 가로 방향

    return y_pred

for lr in learning_rates:
    for rs in reg_strengths:
        # 매 실험마다 가중치 초기화
        cur_W = np.random.randn(D,C) * 0.01
        cur_b = np.random.randn(1, C)

        for i in range(num_iters):
            # mini batch 생성
            batch_indices = np.random.choice(N, batch_size)
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            # X 대신 X batch를 삽입 후 계산
            # 점수 계산
            scores = X_batch.dot(cur_W) + cur_b
            
            # loss, margin 계산
            correct_class_scores = scores[np.arange(batch_size), y_batch].reshape(batch_size,1)
            margins = np.maximum(0, scores - correct_class_scores + 1)
            margins[np.arange(batch_size), y_batch] = 0
            
            # rs(현재 루프의 규제강도) 사용
            data_loss = np.sum(margins) / batch_size
            reg_loss = 0.5 * reg * np.sum(cur_W * cur_W) # L2 규제: 0.5 * lambda * (W^2)
            loss = data_loss + reg_loss

            # gradient 계산
            binary = (margins > 0).astype(float)
            row_sum = np.sum(binary, axis=1)
            binary[np.arange(batch_size), y_batch] = - row_sum
            dW = X_batch.T.dot(binary) / batch_size
            dW += rs * cur_W # L2 규제의 미분값: lambda * W 을 더함
            db = np.sum(binary, axis=0) / batch_size # b에 대한 미분 추가

            # parameter update
            cur_W -=lr * dW
            cur_b -= lr * db

            # 500번마다 출력 <- 최고 성능의 하이퍼파라미터 찾아낼 때 100번마다는 너무 길어짐. 500으로 대체
            if i % 500 == 0:
                # 학습된 W, b로 예측
                y_train_pred = predict(X_batch, W, b)
                train_acc = np.mean(y_train_pred == y_batch) * 100
                
                print(f"Iteration {i}: Loss는 {loss:.4f}, Train Acc: {train_acc:2f}%")
        val_pred = predict(X_val, cur_W, cur_b)
        val_acc = np.mean(val_pred == y_val) * 100

        print(f"lr {lr} / reg {rs} -> val_acc: {val_acc:2f}%")

        if val_acc > best_val:
            best_val = val_acc
            best_W, best_b = cur_W.copy(), cur_b.copy() # copy()를 사용하여 주소가 아닌 값 자체 저장

print(f"\n최고 검증 정확도: {best_val:2f}%")

"""
1차 optimization 후 출력된 결과

Iteration 0: Loss는 12.8662
Iteration 100: Loss는 11.0739
Iteration 200: Loss는 8.7107
Iteration 300: Loss는 9.3315 <- 랜덤한 200장을 사용하기 때문에 fluctuation 발생(매끄럽게 내려가지 않고 일부 loss가 증가하는 부분 존재)
Iteration 400: Loss는 8.8116
Iteration 500: Loss는 8.6338
Iteration 600: Loss는 8.4872
Iteration 700: Loss는 8.0242
Iteration 800: Loss는 8.6124 <- fluctuation
Iteration 900: Loss는 8.1947
Iteration 1000: Loss는 8.2904 <- fluctuation
Iteration 1100: Loss는 8.1228
Iteration 1200: Loss는 7.7424
Iteration 1300: Loss는 7.6628
Iteration 1400: Loss는 7.9045 <- fluctuation
"""
"""
하이퍼파라미터 튜닝 후 출력된 결과

Iteration 0: Loss는 12.6111, Train Acc: 9.500000%
Iteration 500: Loss는 5.7997, Train Acc: 11.000000%
Iteration 1000: Loss는 5.3568, Train Acc: 12.500000%
lr 0.001 / reg 0.1 -> val_acc: 36.600000%
Iteration 0: Loss는 11.5193, Train Acc: 11.000000%
Iteration 500: Loss는 5.8695, Train Acc: 11.500000%
Iteration 1000: Loss는 5.8490, Train Acc: 9.000000%
lr 0.001 / reg 0.25 -> val_acc: 36.700000%
Iteration 0: Loss는 11.0524, Train Acc: 13.000000%
Iteration 500: Loss는 5.6674, Train Acc: 9.000000%
Iteration 1000: Loss는 5.4058, Train Acc: 12.000000%
lr 0.001 / reg 0.5 -> val_acc: 35.000000%
Iteration 0: Loss는 11.9898, Train Acc: 10.000000%
Iteration 500: Loss는 6.5945, Train Acc: 7.500000%
Iteration 1000: Loss는 6.0370, Train Acc: 8.500000%
lr 0.0005 / reg 0.1 -> val_acc: 33.200000%
Iteration 0: Loss는 13.2100, Train Acc: 8.500000%
Iteration 500: Loss는 6.3203, Train Acc: 13.500000%
Iteration 1000: Loss는 6.4355, Train Acc: 11.500000%
lr 0.0005 / reg 0.25 -> val_acc: 34.400000%
Iteration 0: Loss는 10.3751, Train Acc: 10.500000%
Iteration 500: Loss는 6.2143, Train Acc: 9.500000%
Iteration 1000: Loss는 5.9625, Train Acc: 8.500000%
lr 0.0005 / reg 0.5 -> val_acc: 33.600000%
"""

# ---------------------------------
# 8. 시각화
# ---------------------------------

def visualize_weight(best_W):
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # 가중치를 0~255 사이로 정규화 -> 이미지로 볼 수 있도록 변환
    w_min, w_max = np.min(best_W), np.max(best_W)
    w_img = 255.0 * (best_W.reshape(32, 32, 3, 10) - w_min) / (w_max - w_min)

    plt.figure(figsize=(12, 5))
    for i in range(10):
        plt.subplot(2, 5, i+1) # subplot은 1부터 카운트
        
        # 클래스별 가중치를 이미지로 출력
        plt.imshow(w_img[:, :, :, i].astype('uint8'))
        plt.axis('off')
        plt.title(classes[i])
    save_path = "images/linear_classifier"
    plt.savefig(f"{save_path}/svm_weights.png")
    plt.show()

visualize_weight(best_W)


# ---------------------------------
# 9. 최종 테스트
# ---------------------------------

# 테스트 데이터도 학습 데이터처럼 전처리
X_test_processed = x_test.reshape(x_test.shape[0], -1).astype('float32')
X_test_processed /= 255.0

test_pred = predict(X_test_processed, best_W, best_b)
test_acc = np.mean(test_pred == y_test.flatten()) * 100

print(f"\n최종 테스트 정확도(Test Accuracy): {test_acc:.2f}%")
