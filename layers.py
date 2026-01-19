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
    N, C, H, W = x_shape

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
