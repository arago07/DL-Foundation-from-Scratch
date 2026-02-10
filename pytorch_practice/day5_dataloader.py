import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn

# ------------------------------------
# 데이터 준비(Day4과 동일한 가상 데이터)
# ------------------------------------
x_train = torch.randn(100, 3)
true_w = torch.tensor([[2.0], [3.0], [4.0]])
true_b = 5.0
y_train = torch.matmul(x_train, true_w) + true_b + torch.randn(100, 1) * 0.1

print('-' * 60)
print("\nOriginal Data Shape")
print(f" - x_train: {x_train.shape}")
print(f" - y_train: {y_train.shape}\n")
print('-' * 60)

# -------------------------------
# Dataset class 정의
# -------------------------------
class CustomDataset(Dataset):
    def __init__(self, x_data, y_data):
        # data를 클래스 내부에 저장
        self.x = x_data
        self.y = y_data

    def __len__(self):
        # data가 총 몇 개인지 반환
        return len(self.x)
    
    def __getitem__(self, idx):
       # idx(인덱스)에 해당하는 data 1개 반환
       x = self.x[idx]
       y = self.y[idx]
       return x, y
    
# -----------------------------
# DataLoader 연동
# -----------------------------
# 1. dataset 인스턴스 생성
dataset = CustomDataset(x_train, y_train)

# 2. DataLoader 생성
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# --------------------------------
# Full Training Loop
# --------------------------------
print("\nStart Data Loading...")

# 1. 모델, 비용 함수, 옵티마이저 설정
model = nn.Linear(3, 1) # input: 3, output: 1
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

print("\nStart Learning...\n")

# 2. Nested Loop(이중 for문)
for epoch in range(100):
    cost_sum = 0

    for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
        # Weight Update - 5 steps
        prediction = model(batch_x)           # (1) Forward
        loss = criterion(prediction, batch_y) # (2) Loss
        optimizer.zero_grad()                 # (3) Gradient 초기화
        loss.backward()                       # (4) Backward(미분)
        optimizer.step()                      # (5) Weight 수정

        cost_sum += loss.item()
    # 결과 출력 - every 10 epoch
    if (epoch + 1) % 10 == 0:
        avg_cost = cost_sum / len(dataloader) # 평균 손실값 per 1 epoch
        print(f"Epoch {epoch + 1:3d}/100 | Average Loss: {avg_cost:.4f}")



    

