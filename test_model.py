import numpy as np
from cnn import ThreeLayerConvNet

# 1. 가상 데이터 만들기
N = 5
X = np.random.randn(N, 3, 32, 32)
y = np.random.randint(10, size=N) # 정답(0~9 범위, 5개)

# 2. 모델 조립
print("모델 조립 중...")
model = ThreeLayerConvNet(num_filters=32, hidden_dim=100)

# 3. Loss 계산
print("Loss 계산 중...")
loss, grads = model.loss(X, y)

# 4. 결과 확인
print(f"계산된 Loss 값: {loss:.4f}")

# 예상 초기 Loss: 약 2.3(-log(1/10) ≈ 2.3)
if 2.0 < loss < 2.6:
    print("Test passed!")
else:
    print("Cautioin: Loss 값이 예상 범위를 벗어났습니다.")

# Test 결과: 2.3026