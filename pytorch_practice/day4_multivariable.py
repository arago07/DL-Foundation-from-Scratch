import torch
import torch.nn as nn
import torch.optim as optim

# 시드 고정
torch.manual_seed(42)

# multivariable data 생성
x_train = torch.randn(100, 3)

# W, b, y 생성 - 정답: 2*1 + 3*2 + 4*3 + 5
true_w = torch.tensor([[2.0], [3.0], [4.0]])
true_b = 5.0

y_train = torch.matmul(x_train, true_w) + true_b + torch.randn(100, 1) * 0.5

# x_train, y_train 모양 확인
print(f"\nx_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")

model = nn.Linear(3, 1) # 입력 3개, 출력 1개

# 모델 내부 가중치 확인
print("\nInitial Weights: ", model.weight)
print("Initial Bias: ", model.bias)

# --------------------------------------------
# Multivariable Learning Loop
# --------------------------------------------
# 1. optimizer setting
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

print('-'*20)
print("\n학습 시작...")

# 2. 학습 루프
epochs = 2000
for epoch in range(epochs):
    # (1) Forward
    prediction = model(x_train)

    # (2) Loss
    loss = criterion(prediction, y_train)

    # (3) Initialization
    optimizer.zero_grad()

    # (4) Backward
    loss.backward()

    # (5) Step
    optimizer.step()

    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

# 3. Result
print("\n학습 완료")
print(f"Answer W: [2.0, 3.0, 4.0], 학습된 W: {model.weight.detach().tolist()}")
print(f"Answer b: 5.0, 학습된 b: {model.bias.item():.4f}")