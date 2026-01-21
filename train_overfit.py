import numpy as np
from cnn import ThreeLayerConvNet

# 1. data 준비
N = 5
X = np.random.randn(N, 3, 32, 32)
y = np.random.randint(10, size=N)

# 2. model 생성
model = ThreeLayerConvNet(weight_scale=1e-2)

# 3. Optimizer
learning_rate = 0.1 # 0.01 -> 0.1
epochs = 20 # 20번 반복 학습

print("학습 시작(초기 Loss 약 2.3 예상)")
print("-" * 30)

# 4. Training Loop
for i in range(epochs):
    # (1) Forward, Backward
    loss, grads = model.loss(X, y)

    # (2) 가중치 업데이트(SGD: Stochastic Gradient Descent)
    # W = W - learning_rate * Gradient
    for param_name in model.params:
        model.params[param_name] -= learning_rate * grads[param_name]
        
    # 로그 출력
    print(f"Epoch {i+1}/{epochs} | Loss: {loss:.4f}") # (진행률) | (현재 성적)

print("-" * 30)
if loss < 0.1:
    print("Succeed")
else:
    print("Loss가 충분히 줄어들지 않았습니다(Hint: 학습률 조절)")

# 1차 실험 출력 결과(learning rate: 0.01)
"""
학습 시작(초기 Loss 약 2.3 예상)
------------------------------
Epoch 1/20 | Loss: 2.2988
Epoch 2/20 | Loss: 2.2910
Epoch 3/20 | Loss: 2.2832
Epoch 4/20 | Loss: 2.2751
Epoch 5/20 | Loss: 2.2668
Epoch 6/20 | Loss: 2.2582
Epoch 7/20 | Loss: 2.2491
Epoch 8/20 | Loss: 2.2396
Epoch 9/20 | Loss: 2.2295
Epoch 10/20 | Loss: 2.2186
Epoch 11/20 | Loss: 2.2069
Epoch 12/20 | Loss: 2.1942
Epoch 13/20 | Loss: 2.1803
Epoch 14/20 | Loss: 2.1650
Epoch 15/20 | Loss: 2.1481
Epoch 16/20 | Loss: 2.1294
Epoch 17/20 | Loss: 2.1086
Epoch 18/20 | Loss: 2.0854
Epoch 19/20 | Loss: 2.0596
Epoch 20/20 | Loss: 2.0307
-> 너무 조금씩 줄어듦
-> 학습률 조절이 필요하다고 판단
"""
# 2차 실험 출력 결과(learning rate; 0.1)
"""
Epoch 1/20 | Loss: 2.2978
Epoch 2/20 | Loss: 2.2411
Epoch 3/20 | Loss: 2.1658
Epoch 4/20 | Loss: 2.0338
Epoch 5/20 | Loss: 1.7953
Epoch 6/20 | Loss: 1.4760
Epoch 7/20 | Loss: 1.2412
Epoch 8/20 | Loss: 1.0397
Epoch 9/20 | Loss: 0.8445
Epoch 10/20 | Loss: 0.8653
Epoch 11/20 | Loss: 1.6358 <- (overshooting)
Epoch 12/20 | Loss: 1.9397 <- (overshooting)
Epoch 13/20 | Loss: 1.1745
Epoch 14/20 | Loss: 0.5666
Epoch 15/20 | Loss: 0.1523
Epoch 16/20 | Loss: 0.0674
Epoch 17/20 | Loss: 0.0452
Epoch 18/20 | Loss: 0.0340
Epoch 19/20 | Loss: 0.0271
Epoch 20/20 | Loss: 0.0225
------------------------------
Succeed
"""