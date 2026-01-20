import numpy as np

def conv_forward_naive(x, w, b, conv_param):
    """
    conv_forward_naive의 Docstring
    
    :param x: 입력 데이터(N, C, H, W)
    :param w: 필터(F, C, HH, WW)
    :param b: 편향
    :param conv_param: 'stride', 'pad' 정보를 답은 딕셔너리
    """
    stride = conv_param['stride']
    pad = conv_param['pad']

    N, C, H, W = x.shape
    F, C, HH, WW = w.shape

    # 1. 출력 크기 계산
    H_out = int(1 + (H - HH + 2 * pad) / stride)
    W_out = int(1 + (W - WW + 2 * pad) / stride)

    # 결과를 담을 'out' 초기화
    out = np.zeros((N, F, H_out, W_out))

    # 2. 패딩 추가
    x_pad = np.pad(x, ((0,0), (0,0), (pad, pad), (pad, pad)), 'constant') # N, C에는 패딩 추가x
    
    # 3. 4중 루프(원활한 논리 이해를 위해 최적화x, for 사용함)
    for n in range(N):                # (1) 각 이미지마다
        for f in range(F):            # (2) 각 필터마다
            for i in range(H_out):    # (3) 세로로 이동하면서
                for j in range(W_out):# (4) 가로로 이동하면서

                    # 시작점과 끝점 계산
                    vert_start = i * stride
                    vert_end = vert_start + HH
                    horiz_start = j * stride
                    horiz_end = horiz_start + WW

                    # 이미지 조각(slice) 가져오기
                    x_slice = x_pad[n, :, vert_start:vert_end, horiz_start:horiz_end]

                    # 연산: (이미지 조각 * 필터)의 합 + bias
                    out[n, f, i, j] = np.sum(x_slice * w[f]) + b[f]


    cache = (x, w, b, conv_param) # 나중에 역전파 사용시 활용
    return out, cache

# -----------------------------------------------
# 테스트 1: 모양
# -----------------------------------------------
print("--- Test 1: Shape Check ---")
x_shape = np.random.randn(2, 3, 10, 10) # 2 pages, 10x10 size
w_shape = np.random.randn(5, 3, 3, 3) # 5 filters, 3x3 size
b_shape = np.random.randn(5)
params = {'pad' : 1, 'stride' : 1}

out, _ = conv_forward_naive(x_shape, w_shape, b_shape, params) # cache 사용 x

print(f"입력 형태: {x_shape.shape}")
print(f"출력 형태: {out.shape}")

if out.shape == (2, 5, 10, 10):
    print('Passed Test 1')
else:
    print("Failed Test 1: Check the formula")

# ------------------------------------------------
# 테스트 2: 계산값
# ------------------------------------------------
print("\n--- Test 2: Value Check ---")

# 모든 값이 1인 3x3 image, filter
x_ones = np.ones((1, 1, 3, 3))
w_ones = np.ones((1, 1, 3, 3))
b_zeros = np.zeros(1)
params_ones = {'pad': 0, 'stride': 1}

out_ones, _ = conv_forward_naive(x_ones, w_ones, b_zeros, params_ones)
result_val = out_ones[0, 0, 0, 0] # 첫 번째 값

print(f"Result: {result_val}")

if np.isclose(result_val, 9.0):
    print("Passed Test 2")
else:
    print("Failed Test 2: (기대값: 9.0, 실제 값: {result_val})")

# -------------------------------------------
# Pooling
# -------------------------------------------
def max_pool_forward_naive(x, pool_param):
    """
    max_pool_forward_naive의 Docstring
    
    :param x: 입력(N, C, H, W)
    :param pool_params: {'pool_height', 'pool_width', 'stride'}
    """

    HH = pool_param['pool_height']
    WW = pool_param['pool_width']
    stride = pool_param['stride']
    N, C, H, W = x.shape

    H_out = int(1 + (H - HH) / stride) # bias x
    W_out = int(1 + (W - WW) / stride)

    out = np.zeros((N, C, H_out, W_out))

    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    # slicing 범위 계산
                    v_start, v_end = i * stride, i * stride + HH
                    h_start, h_end = j * stride, j * stride + WW

                    x_slice = x[n, c, v_start:v_end, h_start:h_end]
                    out[n, c, i, j] = np.max(x_slice)
    cache = (x, pool_param)
    return out, cache

# ----------------------------------
# ReLU
# ----------------------------------
def relu_forward(x):
    """
    relu_forward의 Docstring
    
    :x(입력)에서 0보다 큰 값은 그대로, 0 이하는 0으로
    """
    out = np.maximum(0, x)
    cache = x
    return out, cache

# ----------------------------------
# Affine Forward(행렬 곱)
# ----------------------------------
def affine_forward(x, w, b):
    """
    affine_forward의 Docstring
    
    :param x: 입력 데이터(N, d_1, ..., d_k) -> flatten 일어남
    :param w: 가중치(D, M)
    :param b: 편향(M,)
    """
    # 1. 2차원 행렬로 flatten
    N = x.shape[0]
    x_reshaped = x.reshape(N, -1)

    # 2. 행렬 곱셈 + bias
    out = np.dot(x_reshaped, w) + b

    cache = (x, w, b)
    return out, cache

# ------------------------------------
# Softmax Loss(최종 점수 계산)
# ------------------------------------
def softmax_loss(x, y):
    """
    softmax_loss의 Docstring
    
    :param x: 예측 점수(N, C)
    :param y: 정답 레이블(N,)
    """
    # 최댓값 빼기 >> 오버플로우 방지
    shifted_logits = x - np.max(x, axis=1, keepdims=True) # 이후의 연산을 위해 행렬 모양 유지
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True) # Z = 분모
    log_probs = shifted_logits - np.log(Z) # 로그 안 나눗셈 -> (로그1) - (로그2)

    # N개의 데이터에 대한 확률값 중에서 정답에 해당하는 것(y)만 골라내기
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N # fancy indexing

    # 역전파를 위한 기울기 계산(dout)
    probs = np.exp(log_probs)
    dout = probs.copy() # dout = 'Derivative of Output'
    dout[np.arange(N), y] -= 1 # 정답 클래스를 음수로 만들기 -> 이후 점수를 더 올리려는 행동을 하도록
    dout /= N

    return loss, dout