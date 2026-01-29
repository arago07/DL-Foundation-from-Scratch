import numpy as np


def batchnorm_forward(x, gamma, beta, bn_param):
    # 초기화
    mode = bn_param['mode'] # 'train' or 'test'
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, C, H, W = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(C))
    running_var = bn_param.get('running_var', np.ones(C))

    if mode == 'train':
        # 1. 평균 계산 (axis=(0, 2, 3))
        sample_mean = np.mean(x, axis=(0, 2, 3))

        # 2. 분산 계산
        sample_var = np.var(x, axis=(0, 2, 3))

        # 3. 정규화 및 가중치 적용
        x_normalized = (x - sample_mean.reshape(1, C, 1, 1)) / np.sqrt(sample_var.reshape(1, C, 1, 1) + eps)
        out = gamma.reshape(1, C, 1, 1) * x_normalized + beta.reshape(1, C, 1, 1)

        # 4. Running stats update
        bn_param['running_mean'] = momentum * running_mean + (1 - momentum) * sample_mean
        bn_param['running_var'] = momentum * running_var + (1 - momentum) * sample_var

        # 5. 역전파를 위해 캐시 저장
        cache = (x, x_normalized, sample_mean, sample_var, gamma, beta, eps)
    
    elif mode == 'test':
        # train 때 저장한 running_mean & running_var 사용
        x_normalized = (x - running_mean.reshape(1, C, 1, 1)) / np.sqrt(running_var.reshape(1, C, 1, 1) + eps)
        out = gamma.reshape(1, C, 1, 1) * x_normalized + beta.reshape(1, C, 1, 1)
        cache = None

    return out, cache

def batchnorm_backward(dout, cache):

    # cache 꺼내기
    x, x_normalized, mu, var, gamma, beta, eps = cache
    N, C, H, W = x.shape
    # 사용한 전체 픽셀 수
    M = N * H * W

    # Step 1: 간단한 미분 - beta, gamma
    dbeta = np.sum(dout, axis=(0, 2, 3))
    dgamma = np.sum(dout * x_normalized, axis=(0, 2, 3))

    # Step 2: 중간 미분 - dhatx, dvar, dmu

    # dhatx: 정규화된 x(hatx)에 대한 기울기
    # 식: dout * gamma
    dhatx = dout * gamma.reshape(1, C, 1, 1) # gamma가 현재 (3,) 형태이기 때문
    
    # dvar
    # 식: sum(dhatx * (x - mu) * (-0.5) * (var + eps)^(-1.5))
    dvar = np.sum(dhatx * (x - mu.reshape(1, C, 1, 1)) * (-0.5) * ((var + eps)**(-1.5)).reshape(1, C, 1, 1), axis=(0, 2, 3) )

    # dmu
    # 식: sum(dhatx * (-1)/sqrt) + dvar * sum(-2(x - mu)) / M
    inv_std = 1.0 / np.sqrt(var + eps)
    dmu_path1 = np.sum(dhatx * -inv_std.reshape(1, C, 1, 1), axis=(0, 2, 3))
    dmu_path2 = dvar * np.sum(-2 * (x - mu.reshape(1, C, 1, 1)), axis=(0, 2, 3)) / M
    dmu = dmu_path1 + dmu_path2
    
    # Step 3: dx - (최종) 입력 기울기 구하기
    # 식: (dhatx / sqrt(var + eps)) + (dvar * 2(x-mu) / M) + (dmu / M)
    
    # path1: 정규화 식을 직접 경유
    dx1 = dhatx * inv_std.reshape(1, C, 1, 1)
    # path2: 분산 경유
    dx2 = dvar.reshape(1, C, 1, 1) * 2 * (x - mu.reshape(1, C, 1, 1)) / M
    # path3: 평균 경유
    dx3 = dmu.reshape(1, C,  1, 1) / M
    
    dx = dx1 + dx2 + dx3

    return dx, dgamma, dbeta


# ================================================
# Test Code - Main Test: 수치미분과 비교
# ================================================
def rel_error(x, y):
    # 두 행렬 간의 상대 오차를 계산(작아야 좋음)
    return np.max(np.abs(x-y) / np.maximum(1e-8, np.abs(x) + np.abs(y)))

def eval_numerical_gradient_array(f, x, df, h=1e-5):
    # 수치 미분 계산 함수(df: 상류 미분값)
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite']) # nditer: 다차원 배열을 한 줄로 -> 차례로 방문
    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        x[ix] = oldval + h # h가 1e-5인 이유: '아주 작은 수'를 더했다 빼기 위해 -> 중앙 차분 계산
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval

        # (f(x+h) - f(x-h)) / 2h * dout
        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad

# data 준비
np.random.seed(231) # 랜덤 숫자를 고정
N, C, H, W = 2, 3, 4, 4
x = np.random.randn(N, C, H, W)
gamma = np.random.randn(C)
beta = np.random.randn(C)
dout = np.random.randn(N, C, H, W) # 상류에서 내려온 미분값을 해당 랜덤값으로 가정

bn_param = {'mode': 'train'}

# Forward 실행
out, cache = batchnorm_forward(x, gamma, beta, bn_param)

# Barkward 실행 <- 검증할 코드
dx, dgamma, dbeta = batchnorm_backward(dout, cache)

# 수치미분과 비교
# 1. gamma의 경우
fx = lambda g: batchnorm_forward(x, g, beta, bn_param)[0]
dgamma_num = eval_numerical_gradient_array(fx, gamma, dout)

# 2. beta의 경우
fx = lambda b: batchnorm_forward(x, gamma, b, bn_param)[0]
dbeta_num = eval_numerical_gradient_array(fx, beta, dout)

# 3. input(X)의 ㄱㅇ우
fx = lambda v: batchnorm_forward(v, gamma, beta, bn_param)[0]
dx_num = eval_numerical_gradient_array(fx, x, dout)

# 결과 출력
print('dx error: ', rel_error(dx_num, dx))
print('dgamma error: ', rel_error(dgamma_num, dgamma))
print('dbeta error: ', rel_error(dbeta_num, dbeta))

print('-' * 30)
if rel_error(dx_num, dx) < 1e-7:
    print("backward 구현 성공(오차가 거의 없습니다.)")
else:
    print("backward 구현 실패. 오차 확인 필요")

# 출력 결과
"""
dx error:  2.2531790046217765e-09
dgamma error:  2.6497517070987826e-11
dbeta error:  3.275567716960339e-12
------------------------------
backward 구현 성공(오차가 거의 없습니다.)
"""