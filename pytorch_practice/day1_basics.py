import torch
import numpy as np

# 장치 설정 확인
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"현재 사용 장치: {device}")

# 1. 생성 & 변환(Numpy <-> PyTorch)
np_data = np.array([[1, 2], [3, 4]]) # 기존(numpy) 방식
tensor_data = torch.from_numpy(np_data) # (numpy -> pytorch) 변환
print(f"\n1. Tensor 변환 완료:\n{tensor_data}")
back_to_numpy = tensor_data.numpy() # (pytorch -> numpy)로 재변환

# 2. 모양 바꾸기(reshape 대신 .view 사용)
mock_image = torch.randn(2, 3, 4, 4) # (N, C, H, W)
print(f"\n2. 원본 이미지 모양: {mock_image.shape}") # 원본 이미지 모양 출력
flattened = mock_image.view(2, -1) # (N, C*H*W)로 flatten
print(f"\n   펼친 모양: {flattened.shape}")

# 3. 장치 이동(CPU -> GPU)
tensor_gpu = tensor_data.to(device)
print(f"\n3. GPU로 이동 완료: {tensor_gpu.device}")
print(f"\n   다시 CPU로 복귀: {tensor_gpu.cpu().numpy()}") # GPU -> CPU -> Numpy여야 함

# 4. 행렬 곱셈(np.dot을 대체)
A = torch.randn(2, 3).to(device)
B = torch.randn(3, 4).to(device)

# numpy: np.dot(A, B)
# pytorch: torch.mm(A, B) or A @ B
result = torch.mm(A, B)
print(f"\n4. 행렬 곱셈 결과 모양: {result.shape}") # 예상: (2, 4)
