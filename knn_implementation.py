import numpy as np
import time

# -----------------------------------------------------------
# 1. 가상 데이터 생성 (N= 훈련 데이터 수, M= 테스트 데이터 수, D= 차원)
# -----------------------------------------------------------
N = 5000
M = 500
D = 3072

X_train = np.random.rand(N, D)
X_test = np.random.rand(M, D)

print(f"훈련 데이터 크기: {X_train}")
print(f"테스트 데이터 크기: {X_test}")

# -----------------------------------------------------
# 2. Two loops 구현 - 가장 느린 방식(예상)
# -----------------------------------------------------
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
# ----------------------------------------------

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

# ----------------------------------------
# 4. k-NN 찾기
# ----------------------------------------
# 설정
K = 5

# dists_no_loop: (500, 5000) 행렬
# axis = 1: 각 행 안에서 5000개의 거리를 서로 비교하는 것이므로 axis는 1.

# 1) 시작 시간 기록
start_time = time.time()

# 2) 정렬된 인덱스 얻기 - 거리가 작은 순서대로 인덱스 나열
# 중요 - 작은 거리의 위치(인덱스). 거리 자체가 중요한 것이 아님 -> sort가 아닌 argsort 사용
# 결과의 shape: (500, 5000)
sorted_indices = np.argsort(distance_no_loop, axis=1) # indices = index의 복수형 단어

# 3) 앞에서부터 k개 자르기
# 모든 행에 대해서 0 ~ K-1번째 열까지 가져오기
nearest_k_indices = sorted_indices[:, :K]

# 4) 종료 시간 기록
end_time = time.time()

# 5) 결과 출력
print(f"\n최근접 이웃 인덱스 추출 완료. 소요 시간: {end_time - start_time:.4f}sec")
print(f"결과 행렬 크기: {nearest_k_indices.shape}") # (500, 5) 예상
print(f"첫 번째 테스트 샘플의 이웃 5명 번호: {nearest_k_indices[0]}")
print(f"두 번째 테스트 샘플의 이웃 5명 번호: {nearest_k_indices[1]}")

# result:
# 최근접 이웃 인덱스 추출 완료. 소요 시간: 0.1096sec
# 결과 행렬 크기: (500, 5)
# 첫 번째 테스트 샘플의 이웃 5명 번호: [ 529 1345 2392  290 1834]
# 두 번째 테스트 샘플의 이웃 5명 번호: [1630  645 1538 1704 4902]

# ------------------------------------------------------
# 5. voting - 예측
# ------------------------------------------------------
num_test = M

# 1) 실제 데이터셋이 없으므로 테스트를 위해 random한 정답지를 생성
if 'y_train' not in locals():
    y_train = np.random.randint(0, 10, N)

# 2) 최종 예측값을 저장할 배열 생성
y_pred = np.zeros(num_test, dtype=int)

# 3) 각 테스트 샘플에 대해 vote - 벡터화가 까다로워 일반적으로 루프를 사용. 자세한 이유는 Notion에 정리
for i in range(num_test):
    # (1) i번째 테스트 데이터와 가장 가까운 5명의 레이블 가져오기
    closest_y = y_train[nearest_k_indices[i]]

    # (2) 5개의 이름 중에 가장 많이 나온 이름에 투표(voting)
    counts = np.bincount(closest_y)

    # (3) 가장 많이 나온 인덱스를 최종 예측값으로 결정
    y_pred[i] = np.argmax(counts)

# 4) 결과 출력
print(f"\n[Voting] 완료. 가장 많이 나온 인덱스 예측 결과: {y_pred[:5]}")

# result: [0 0 0 4 0]
# 해당 result의 의미: 1, 2, 3, 5번째 데이터는 0번 카테고리로 예측, 4번째는 4번 카테고리로 예측

# 0이 많이 보이는 이유?
# 1. 0~9중에서 무작위로 뽑다 보면 숫자가 몰려서 나올 수 있음
# 2. X_train, X_test는 random한 숫자 뭉치로, 그 레이블도 랜덤이기 때문