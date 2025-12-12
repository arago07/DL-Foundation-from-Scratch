import numpy as np
import time

# 1. 가상 데이터 생성 (N= 훈련 데이터 수, M= 테스트 데이터 수, D= 차원)
N = 5000
M = 500
D = 3072

X_train = np.random.rand(N, D)
X_test = np.random.rand(M, D)

print(f"훈련 데이터 크기: {X_train}")
print(f"테스트 데이터 크기: {X_test}")

# -----------------------------------------
# 2. Two loops 구현 - 가장 느린 방식(예상)
distance_two_loops = np.zeros((M, N)) # M x N 거리 행렬을 초기화
start_time = time.time()

for i in range(M):
    for j in range(N):
        # euclidian distance의 제곱 : (i번째 행의 전체 열 - j번째 행의 전체 열)의 제곱
        sum_squares = np.sum((X_test[i, :] - X_train[j, :]) ** 2)

        distance_two_loops[i,j] = np.sqrt(sum_squares)

end_time = time.time() # 끝나는 시간 기록
print(f"\n[Two loops] 소요 시간: {end_time - start_time:.4f}") # 소숫점 넷째 자리까지만 표시

# result: 9.4559 sec

# ----------------------------------------------

# 3. No loop(부분 벡터화) 구현 - 약간 빨라진 방식(예상)
# (a-b)**2 = a**2 -2*a*b + b**2 이용

distance_no_loop = np.zeros((M, N)) # 행렬 초기화
start_time = time.time()

# 1) 테스트 데이터 각 행의 제곱합 계산 (M x 1)
# X_test는 (500, 3072) -> 제곱, axis= 1로 합산 -> (500,) 형태
sum_sq_test = np.sum(X_test**2, axis=1)

# 2) 훈련 데이터 각 행의 제곱합 계산 (N x 1)
# X_train은 (5000, 3072) -> 제곱, axis= 1로 합산 -> (5000,) 형태
sum_sq_train = np.sum(X_train**2, axis=1)

# 3) 중요: 행렬 곱셈
dot_product = np.dot(X_test, X_train.T)

# 4) 3개 항 전부 더하기(브로드캐스팅 사용)
distance_no_loop_sq = sum_sq_test[:, np.newaxis] - 2 * dot_product + sum_sq_train[np.newaxis, :]

# 5) Two loops와의 공정성을 위해 np.sqrt를 적용
distance_no_loop = np.sqrt(distance_no_loop_sq)

end_time = time.time()
print(f"\n[No loops] 소요 시간: {end_time - start_time:.4f}")
# result: 0.0808 sec <- 압도적인 속도 차이. numpy의 효율성

# -----------------------------------------------------


