import torch
import torch.nn as nn
import torch.optim as optim

# XOR data 준비
X = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = torch.FloatTensor([[0], [1], [1], [0]])

print(f"XOR Inputs:\n{X}")
print(f"XOR Labels:\n{Y}")

# -------------------------------------------
# MLP 설계: Standard class definition(복잡한 모델)
# -------------------------------------------
"""
class MultiLayerPerceptronV1(nn.Module):
    def __init__(self):
        super().__init__()

        # 1층: 입력 2 -> 은닉층 10(비선형성)
        self.layer1 = nn.Linear(2, 10)
        self.relu = nn.ReLU()
        # 2층: 은닉층 10 -> 출력 1
        self.layer2 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid() # 0~1 사이 확률값 출력

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.sigmoid(out)
        return out
    
model = MultiLayerPerceptronV1()
"""
# --------------------------------------------
# MLP 학습: Simple Sequential 방식(단순 적층 구조)
# --------------------------------------------
class MultiLayerPerceptronV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)
# --------------------------------------------
# XOR model 학습 
# --------------------------------------------
# 1. model, loss function, optimizer 설정
"""
model = MultiLayerPerceptronV1()
"""
model = MultiLayerPerceptronV2()
criterion = nn.BCELoss() # 이진 분류를 위한 binary cross entropy loss
optimizer = optim.SGD(model.parameters(), lr=1) # learning rate를 높게 설정

# 2. Learning Loop(Iteration: 10,000)
for step in range(10001):
    # (1) Forward Pass
    prediction = model(X)

    # (2) Loss 계산
    loss = criterion(prediction, Y)

    # (3) Gradient 초기화
    optimizer.zero_grad()

    # (4) Backward Pass(역전파)
    loss.backward()

    # (5) parameter update
    optimizer.step()

    if step % 1000 == 0:
        print(f"Step: {step:5d} | Loss: {loss.item():.6f}")

# 3. Test
print('-' * 30)
with torch.no_grad():
    hypothesis = model(X)
    predicted = (hypothesis > 0.5).float()
    accuracy = (predicted == Y).float().mean()
    print(f"모델의 예측값:\n{hypothesis.detach().numpy()}") # tensor -> numpy 배열
    print(f"최종 판정(기준: 0.5):\n{predicted.detach().numpy()}")
    print(f"Accuracy: {accuracy.item() * 100}%")