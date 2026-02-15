import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 1. Hyperparameters
training_epochs = 15
batch_size = 100
"""
learning_rate = 0.1 # SGD의 경우
"""
learning_rate = 0.001 # Adam의 경우

# 2. Dataset(MNIST) Loading
mnist_train = dsets.MNIST(root='MNIST_data/', train=True,
                          transform=transforms.ToTensor(), download=True) # data x -> download
mnist_test = dsets.MNIST(root='MNIST_data/', train=False,
                         transform=transforms.ToTensor(), download=True)

# 3. Setting DataLoader (batch 단위)
data_loader = DataLoader(dataset=mnist_train, batch_size=batch_size,
                         shuffle=True, drop_last=True) # 자투리 데이터 버리기
# 4. Model 설계 - MLP
# input: 28 * 28 = 784, output: 10 classes(0~9)
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10) # Softmax: CrossEntropyLoss가 내부적으로 해주므로 Linear() 뒤에 생략
)

# 5. Loss Function & Optimizer
criterion = nn.CrossEntropyLoss() # 내부적으로 LogSoftmax + NLLLoss
"""
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
"""
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 6. Learning Loops
print('-' * 50)
print("Started Learning...\n")
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = len(data_loader)

    for X, Y in data_loader:
        # flatten: (batch_size, 1, 28, 28) -> (batch_size, 784)
        X = X.view(-1, 28 * 28)
        
        optimizer.zero_grad()
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print(f"Epoch: {epoch + 1 :>2} | Cost: {avg_cost:.9f}")

print("\nFinished Learning!")
print('-' * 50)

# 7. Accuracy Test
with torch.no_grad(): # test용이므로 grad 기록이 필요없음
    X_test = mnist_test.data.view(-1, 28 * 28).float()
    Y_test = mnist_test.targets

    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print(f"Accuracy: {accuracy.item() * 100:.2f}%")