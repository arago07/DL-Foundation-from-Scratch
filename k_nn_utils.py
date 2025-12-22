"""
함수 모아두는 전용 파일
"""

import numpy as np
def compute_distances_no_loops(X_test, X_train):
    # 1) 제곱합 계산
    sum_sq_test = np.sum(X_test**2, axis=1)
    sum_sq_train = np.sum(X_train**2, axis=1)
    
    # 2) 행렬 곱셈
    dot_product = np.dot(X_test, X_train.T)
    
    # 3) 거리 계산 (a^2 - 2ab + b^2)
    # np.maximum(dists, 0): 계산 오차로 아주 미세한 음수가 나오는 것 방지
    # 컴퓨터가 실수를 다루는 방식(부동 소수점 연산)의 한계로 오차 발생 가능함
    dists_sq = sum_sq_test[:, np.newaxis] - 2 * dot_product + sum_sq_train[np.newaxis, :]
    dists = np.sqrt(np.maximum(dists_sq, 0))
    
    return dists

def predict_labels(dists, y_train, k=1):
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test, dtype=int)
    
    # 1) 거리가 가까운 순서대로 인덱스 정렬
    sorted_indices = np.argsort(dists, axis=1)
    
    # 2) Voting
    for i in range(num_test):
        # i번째 테스트 데이터의 k-nearest 레이블 추출
        closest_y = y_train[sorted_indices[i, :k]]
        # 투표
        counts = np.bincount(closest_y)
        y_pred[i] = np.argmax(counts)
        
    return y_pred