import matplotlib.pyplot as plt
import os

# 1. 데이터 설정
epochs = list(range(1, 16))
sgd_losses = [0.633756697, 0.243587419, 0.172724292, 0.134236977, 0.109707355, 
              0.091706477, 0.079052761, 0.068245374, 0.060426235, 0.052924395, 
              0.046671707, 0.041743226, 0.037284929, 0.033044375, 0.030020472]

adam_losses = [0.388681740, 0.164332420, 0.112894975, 0.085849963, 0.068568699, 
               0.053884186, 0.043679729, 0.036389608, 0.030057231, 0.024243476, 
               0.021060981, 0.018012578, 0.015414852, 0.013232644, 0.011750864]

# 2. 경로 설정 및 디렉토리 생성
save_dir = "/Users/arago/Desktop/DLFoundation/DL-Foundation-from-Scratch/images/pytorch_practice"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f"Directory created: {save_dir}")

# 3. 그래프 그리기
plt.figure(figsize=(10, 6))
plt.plot(epochs, sgd_losses, label='SGD (lr=0.1)', marker='o', color='gray', linestyle='--')
plt.plot(epochs, adam_losses, label='Adam (lr=0.001)', marker='s', color='blue')

plt.title('Loss Curve Comparison: SGD vs Adam (MNIST)', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss (Cost)', fontsize=12)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)

# 4. 저장
save_path = os.path.join(save_dir, 'mnist_loss_comparison.png')
plt.tight_layout()
plt.savefig(save_path, dpi=300)

print(f"Success! Graph saved at: {save_path}")