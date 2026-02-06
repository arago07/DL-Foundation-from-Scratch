import torch
import torch.nn as nn

# 1. data 준비(x, y)
x = torch.tensor(1.0)
y = torch.tensor(2.0)

# 2. model parameter(w, b)
w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

# 3. Forward
# 예측값: y_hat = w * x + b
y_hat = w * x + b
loss = (y_hat - y)**2

print(f"예측값: {y_hat.item()}, Loss: {loss.item()}")

# 4. Backward
# Today's highlight
loss.backward() # 지금까지의 연산 기록을 이용해 미분값 자동 계산

# 5. Result
print(f"w의 기울기: {w.grad}")
print(f"b의 기울기: {b.grad}")

"""
(Expectation)
Loss = (wx + b - y)^2
d(Loss)/dw = 2(wx + b - y) * x = 2(1 * 1 + 0 - 2) * 1 = -2
w.grad는 -2여야 함

(Result)
예측값: 1.0, Loss: 1.0
w의 기울기: -2.0
b의 기울기: -2.0

-> 일치
"""
# -----------------------------------------------------------
# The Importance of Initializing Derivative Value in PyTorch
print('-' * 20)
print("Trying to Catch Gradient Accumulating")

# play it one more time(Forward -> Backward)
y_hat = w * x + b
loss = (y_hat - y)**2
loss.backward()

print(f"2nd w.grad: {w.grad}")
# Result: - 4.0 ((- 2.0) + (-2.0))
# PyTorch는 기존 grad에 새로운 grad를 더한다(누적)

# Solution
# -> next step으로 넘어가기 전에 반드시 기존 기울기를 0으로 초기화해야 함
w.grad.zero_()
b.grad.zero_()
print(f"초기화 후의 w.grad: {w.grad}")
# Result
# 초기화 후의 w.grad: 0.0

# -------------------------------------------------
# w, b 대신 nn.Linear 사용

model = nn.Linear(1, 1) # 입력 차원 1, 출력 차원 1 -> w 1개, b 1개인 선형 회귀

print('-'*20)
print(f"\n[nn.Linear 내부 확인]")
print(f"Weight: {model.weight}") # 자동적으로 requires_grad=True임
print(f"Bias: {model.bias}")

"""
(Result)
[nn.Linear 내부 확인]

(1st try)
Weight: Parameter containing:
tensor([[0.7379]], requires_grad=True)
Bias: Parameter containing:
tensor([0.3154], requires_grad=True)

(2nd try)
Weight: Parameter containing:
tensor([[-0.1587]], requires_grad=True)
Bias: Parameter containing:
tensor([0.3078], requires_grad=True)
"""