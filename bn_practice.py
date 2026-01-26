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
        print(f"sample_mean 형태: {sample_mean.shape}")
        
        # 2. 분산 계산
        sample_var = np.var(x, axis=(0, 2, 3))
        print(f"sample_var 형태: {sample_var.shape}")

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

# ============================================
# Test Code
# ============================================

x = np.random.randn(4, 3, 8, 8) # batch: 4, channel: 3, H:8, W:8

# 초기화 파라미터 설정(Scale & Shift)
# 채널 3개 -> gamma, beta 3개씩
gamma = np.ones(3) # test고 곱하기를 해야 하므로 모두 1
beta = np.zeros(3)
bn_param = {'mode': 'train'}

# 함수 실행
print("입력 데이터 모양: ", x.shape)
out, cache = batchnorm_forward(x, gamma, beta, bn_param)

# 검증: 모양 유지되는지
print("출력 데이터 모양: ", out.shape)

if x.shape== out.shape:
    print('차원 유지 성공')
else:
    print('차원 유지 실패')