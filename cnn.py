import numpy as np
from layers import *

class ThreeLayerConvNet:
    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0):
        
        self.params = {}
        self.reg = reg
        C, H, W = input_dim

        # 1. Conv Layer(W1): 32x32 image의 크기를 그대로 유지
        # W1 shape: (필터 개수, 채널, 필터 높이, 필너 너비)
        self.params['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
        self.params['b1'] = np.zeros(num_filters)

        # 2. Hidden Affine Layer(W2)
        # Pool(2x2)를 거쳐 32x32 -> 16x16
        pool_output_dim = num_filters * (H // 2) * (W // 2)

        self.params['W2'] = weight_scale * np.random.randn(pool_output_dim, hidden_dim)
        self.params['b2'] = np.zeros(hidden_dim)

        # 3. Output Affine Layer (W3)
        self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b3'] = np.zeros(num_classes)

        # data type 정리
        for k, v in self.params.items():
            self.params[k] = v.astype(np.float64) # 정밀 실수(float)로 세팅
    
    def loss(self, X, y=None):
        """
        loss의 Docstring
        
        :param X: 입력 데이터 (N, C, H, W)
        :param y: 정답 레이블 (N,)
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # filter 크기에 따른 padding 자동 계산
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        # ---------------------------------------------
        # Forward Pass(순전파)
        # ---------------------------------------------
        conv_out, cache_conv = conv_forward_naive(X, W1, b1, conv_param)
        relu1_out, cache_relu1 = relu_forward(conv_out)
        pool_out, cache_pool = max_pool_forward_naive(relu1_out, pool_param)

        # --------------------------------------------
        # Affine - ReLU Layer(Hidden)
        # --------------------------------------------
        affin1_out, cache_affine1 = affine_forward(pool_out, W2, b2)
        relu2_out, cache_relu2 = relu_forward(affin1_out)

        # ---------------------------------------------
        # Affine Layer(Output) -> Scores 계산
        # ---------------------------------------------
        scores, cache_affine2 = affine_forward(relu2_out, W3, b3)

        # 정답(y)이 없으면 점수만 반환(test)
        if y is None:
            return scores
        
        # ---------------------------------------------
        # Loss 계산 (Sofrmax Cross Entropy)
        # ---------------------------------------------
        data_loss, dout = softmax_loss(scores, y)

        # L2 Regularization(가중치 규제): 모델이 너무 복잡해지지 않도록 W 제곱합 더하기
        w_reg_loss = 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))
        loss = data_loss + w_reg_loss

        # --------------------------------------------
        # Backward Pass(역전파)
        # --------------------------------------------
        # 1. Softmax 뒤에서부터 거꾸로
        # loss, dout = softmax_loss(scores, y) <- 위에서 함

        # 2. Output Layer(Affine) 뒤로 가기
        # dout이 계속 업데이트되며 앞으로 전달
        dout, dW3, db3 = affine_backward(dout, cache_affine2)

        # 3. Hidden Layer(ReLU -> Affine) 뒤로 가기
        dout = relu_backward(dout, cache_relu2)
        dout, dW2, db2 = affine_backward(dout, cache_affine1)

        # 4. Conv Layer(Pool -> ReLU -> Conv) 뒤로 가기
        dout = max_pool_backward_naive(dout, cache_pool)
        dout = relu_backward(dout, cache_relu1)
        dx, dW1, db1 = conv_backward_naive(dout, cache_conv)

        # 5. Regulation(가중치 규제) 미분 추가
        # Loss에 (+ 0.5 * reg * W^2)를 함 -> 미분 후 reg * W가 추가
        dW1 += self.reg * W1
        dW2 += self.reg * W2
        dW3 += self.reg * W3

        # 6. 결과를 dictionary에 담기
        grads = {
            'W1': dW1, 'b1': db1,
            'W2': dW2, 'b2': db2,
            'W3': dW3, 'b3': db3
        }

        return loss, grads
    
            