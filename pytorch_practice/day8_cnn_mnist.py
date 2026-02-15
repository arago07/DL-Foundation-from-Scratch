import torch
import torch.nn as nn
import random
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# seed 고정 - 재현성
random.seed(777)
torch.manual_seed(777)

# Setting device - MPS 우선
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
if torch.backends.mps.is_available():
    torch.mps.manual_seed(777)

# Hyperparameter, Data Loading
learning_rate = 0.001
training_epochs = 15
batch_size = 100

mnist_train = dsets.MNIST(root='MNIST_data/', train=True, transform=transforms.ToTensor(), download=True)
mnist_test = dsets.MNIST(root='MNIST_data/', train=False, transform=transforms.ToTensor(), download=True)
data_loader = DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # 1st Feature Extractor(Input: 1x28x28)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), # Output: 32x28x28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)                 # Output: 32x14x14
        )

        # 2nd Feature Extractor
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # Output: 64x14x14
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)                  # Output: 64x7x7
        )

        # Classifier - Fully Connected Layer
        self.fc = nn.Linear(64 * 7 * 7, 10, bias=True) # fc - Fully Connected Layer

        # Xavier Initialization
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1) # tensor를 flatten
        out = self.fc(out)
        return out

# --------------------------------------------
# Version Upgrade
# --------------------------------------------
# 2 layers -> 3 layers
# Batch Normalization
# Dropout

class DeepCNN(nn.Module):
    def __init__(self):
        super(DeepCNN, self).__init__()

        # Layer 1: (1, 28, 28) -> (32, 14, 14)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Layer 2: (32, 14, 14) -> (64, 7, 7)
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Layer 3(Added): (64, 7, 7) -> (128, 3, 3)
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1) # padding - 홀수 크기 대응 위해
        )

        # Classifier(Layer 4)
        self.fc1 = nn.Linear(128 * 4 * 4, 625, bias=True)
        self.layer4 = nn.Sequential(
            self.fc1,
            nn.BatchNorm1d(625),
            nn.ReLU(),
            nn.Dropout(0.5) # 학습 시 뉴런의 50% 무작위로 off
        )
        self.fc2 = nn.Linear(625, 10, bias=True)

        # Initialization(He <- ReLU와 좋은 조합)
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1) # flatten
        out = self.layer4(out) # classifier
        out = self.fc2(out)
        return out 
    
def evaluate_model(model, test_dataset, device):
    model.eval() # 평가 모드
    with torch.no_grad():
        # 데이터 준비
        X_test = test_dataset.data.view(len(test_dataset), 1, 28, 28).float().to(device)/255.0
        Y_test = test_dataset.targets.to(device)

        # 예측
        prediction = model(X_test)
        correct_prediction = torch.argmax(prediction, 1) == Y_test
        accuracy = correct_prediction.float().mean()
        
    return accuracy.item() * 100

# ---------------------------------------------------------
# 공통 학습 및 평가 함수 정의
# ---------------------------------------------------------
def run_experiment(model_class, model_name, device, train_loader, test_dataset):
    print(f"\n{'-' * 20} {model_name} Experiment Start {'-' * 20}")
    
    # model initialization & 장치 이동
    model = model_class().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    cost_history = [] 
    # Training Loop
    total_batch = len(train_loader)
    for epoch in range(training_epochs):
        model.train()
        avg_cost = 0

        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)

            optimizer.zero_grad()
            hypothesis = model(X)
            cost = criterion(hypothesis, Y)
            cost.backward()
            optimizer.step()
            avg_cost += cost / total_batch

        cost_history.append(avg_cost.item())
        print(f'Epoch: {epoch + 1:>4} | cost = {avg_cost:>.9f}')
    
    # 평가 (Evaluation)
    acc = evaluate_model(model, test_dataset, device)
    print(f"\n[{model_name}] Learning Finished!")
    print(f"[{model_name}] Final Accuracy: {acc:.2f}%")
    return {
        'name': model_name,
        'costs': cost_history,
        'acc': acc
    }

#----------------------------------------------------
# Visualization
# ---------------------------------------------------
def visualize_results(result_list):
    save_path = "/Users/arago/Desktop/DLFoundation/DL-Foundation-from-Scratch/images/pytorch_practice"
    
    # 폴더가 없으면 생성 (에러 방지)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.figure(figsize=(12,5))

    # Cost Curve
    plt.subplot(1, 2, 1)
    for res in result_list:
        plt.plot(res['costs'], label=f"{res['name']} (Cost)")
    plt.title('Training Cost')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.legend() # 범례
    plt.grid(True, linestyle='--', alpha=0.6)

    # Accuracy Bar Chart
    plt.subplot(1, 2, 2)
    names = [res['name'] for res in result_list]
    accs = [res['acc'] for res in result_list]
    bars = plt.bar(names, accs, color=['skyblue', 'salmon'], alpha=0.8)

    # gragh 상단에 accuracy 표시
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.title('Final Test Accuracy')
    plt.ylim(min(accs) - 1, 100) # 차이가 눈에 잘 보이도록 최솟값 조정
    plt.ylabel('Accuracy (%)')

    # 파일 저장
    full_file_path = os.path.join(save_path, "cnn_comparison_result.png")
    plt.savefig(full_file_path, dpi=300) # 고해상도
    print(f"\n✅ 그래프가 저장되었습니다: {full_file_path}")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":
    
    exp_results = []

    # Version 1: 기본 CNN
    exp_results.append(run_experiment(CNN, "Basic CNN", device, data_loader, mnist_test))

    # Version 2: Deep CNN (BN + Dropout + 기존 Layers(+1))
    exp_results.append(run_experiment(DeepCNN, "Deep CNN", device, data_loader, mnist_test))

    # Result Printing
    print('\n' + '=' * 50)
    print(f"{'Model Name: ':<15} | {'Accuracy: ':<10}")
    print('-' * 50)

    for res in exp_results:
        name = res['name']
        acc = res['acc']
        print(f"{name:<15} | {acc:.2f}%")
    print('=' * 50)

    visualize_results(exp_results)

