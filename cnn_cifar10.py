import numpy as np
import matplotlib.pyplot as plt
from cnn import ThreeLayerConvNet
from keras.datasets import cifar10

# ------------------------------------------
# 1. data 로드(Lite version: 5,000개 사용)
# ------------------------------------------
print("데이터 로딩 중...(5,000개)")
(X_train_full, y_train_full), (X_test, y_test) = cifar10.load_data()

# data 줄이기(training: 5000, validation: 500)
num_train = 5000
num_val = 500

# (특정 부분만 골라내는 역할인) mask의 범위
mask_train = range(num_train)
mask_val = range(num_train, num_train + num_val)

X_train = X_train_full[mask_train]
y_train = y_train_full[mask_train].flatten() # 1차원으로 flatten

X_val = X_train_full[mask_val]
y_val = y_train_full[mask_val].flatten()

# ------------------------------------------
# 2. 전처리(Processing)
# -------------------------------------------
# (1) float32로 변환(사용환경에 따른 메모리 절약을 위해)
X_train = X_train.astype(np.float32)
X_val = X_val.astype(np.float32)

# (2) Mean Subtraction(평균 빼기)
# 이미지의 중심을 0으로 맞추어 학습을 원활하게 하기 위해
mean_image = np.mean(X_train, axis=0) # 평균
X_train -= mean_image
X_val -= mean_image

# (3) 채널 위치 변경(Keras: HWC -> 이 모델: CHW)
# (N, 32, 32, 3) -> (N, 3, 32, 32)
X_train = X_train.transpose(0, 3, 1, 2)
X_val = X_val.transpose(0, 3, 1, 2)

print(f"학습 데이터의 형상 확인: {X_train.shape}") # (5000, 3, 32, 32) 예상
print("-" * 30)

# ---------------------------------------------
# 3. 모델 생성 & 학습 설정
# ---------------------------------------------
model = ThreeLayerConvNet(input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                          num_classes=10, weight_scale=1e-3,reg=0.001)

# 학습 파라미터
batch_size = 50 # 현재 RAM의 크기 고려
epochs = 5
iterations_per_epoch = max(num_train // batch_size, 1) # 안전장치로 max(...,1)
total_iters = iterations_per_epoch * epochs

learning_rate = 0.01
loss_history = []
train_acc_history = []
val_acc_history = []

print(f"총 반복 횟수: {total_iters}")

# -----------------------------------------
# 4. Training Loop(학습 루프)
# -----------------------------------------
for it in range(total_iters):
    # (1) mini batch
    batch_mask = np.random.choice(num_train, batch_size)
    X_batch = X_train[batch_mask]
    y_batch = y_train[batch_mask]

    # (2) 기울기 계산 & 업데이트
    loss, grads = model.loss(X_batch, y_batch)
    loss_history.append(loss)

    for param_name in model.params:
        model.params[param_name] -= learning_rate * grads[param_name]

    # (3) (매 epoch마다) 로그를 출력
    if it % 10 == 0: # 10번마다 생존 신고해라!
        print(f"Iter: {it}/{total_iters} | Loss: {loss:.4f}")

# ------------------------------------------
# 5. Visualizing Filter
# ------------------------------------------
# W1 가져오기: (32, 3, 7, 7)
W1 = model.params['W1']
W1 = W1.transpose(0, 2, 3, 1)

# 0~255 사이로 스케일링(이미지화를 위해)
W1_min, W1_max = np.min(W1), np.max(W1)
W1 = 255.0 * (W1 - W1_min) / (W1_max - W1_min)
W1 = W1.astype('uint8')

plt.figure(figsize=(8, 8))
for i in range(32): # 32개의 필터 전부 그리기
    plt.subplot(4, 8, i+1)
    plt.imshow(W1[i])
    plt.axis('off')
    plt.title(f'{i+1}')

plt.savefig('images/cnn/filter_visualization.png')
print("필터 시각화 저장 완료: images/cnn/filter_visualization.png")
plt.show()