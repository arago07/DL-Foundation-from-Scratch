import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 장치 설정
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print('-'*20)
print(f"Using device: {device}")

# --------------------------------------
# Generating Synthetic Data
# --------------------------------------
# Data: x는 [0, 10]인 숫자 100개
x_train = torch.linspace(0, 10, 100).unsqueeze(1).to(device) 

# 정답(y) 생성: y = 2x + 1
y_train = 2 * x_train + 1 + torch.randn(100, 1).to(device) # 약간의 노이즈 추가

# ---------------------------------------
# Declaring Model, MSE, Optimizer
# ---------------------------------------
model = nn.Linear(1, 1).to(device)

# 손실 함수 - MSE(Mean Squared Error: 예측값과 정답 간의 거리를 제곱 후 평균
criterion = nn.MSELoss()

# Optimizer - 점진적 경사 하강법: Stochastic Gradient Descent
# optimizer = optim.SGD(model.parameters(), lr = 0.8): High lr
# optimizer = optim.SGD(model.parameters(), lr = 0.0001): Low lr 
optimizer = optim.SGD(model.parameters(), lr = 0.01)

print("\n[학습 전 Model Parameters]")
print(f"Weight: {model.weight.item():.4f}")
print(f"Bias: {model.bias.item():.4f}")
print('-'*20)

# ------------------------------------------
# Training Loop 구현
# ------------------------------------------
epochs = 200

print("\nStarted Learning...")

for epoch in range(epochs):
    # 1. Forward Pass
    prediction = model(x_train)
    
    # 2. Calculating Loss
    loss = criterion(prediction, y_train)

    # 3. Backward Pass(gradient 계산)
    optimizer.zero_grad() # 지난 기록 초기화
    loss.backward()

    # 4. Parameter Update
    optimizer.step() # 계산된 기울기를 사용해 w, b를 1 step 이동

    # 10번마다 1번씩 학습 현황을 출력
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1} - Loss: {loss.item():.4f},\
              Weight: {model.weight.item():.4f}, Bias: {model.bias.item():.4f}")
              

print("\n학습 완료!")
print(f"Final Weight: {model.weight.item():.4f}")# 예상: 2.0
print(f"Final Bias: {model.bias.item():.4f}")# 예상: 1.0

# ------------------------------------
# Visualization
# ------------------------------------
# 시각화를 위해 Tensor -> NumPy
predicted = model(x_train).detach().cpu().numpy()
x_plot = x_train.cpu().numpy()
y_plot = y_train.cpu().numpy()

plt.figure(figsize=(10, 6))
plt.scatter(x_plot, y_plot, label='Original Data (with noise)', alpha=0.5) # 산점도
plt.plot(x_plot, predicted, color='red', label='Fitted Line (Model Prediction)') # 선 그래프
plt.legend()
plt.title('Linear Regression Result')
plt.show()
