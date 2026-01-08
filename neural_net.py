import numpy as np
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt

class TwoLayerNet:
    """
    A two-layer fully connected neural network.
    Structure: Input - Hidden(ReLU) - Output(Softmax)
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        가중치를 초기화
        std: 가중치가 지나치게 커지지 않도록 제약을 거는 스케일 값(보통 0.0001)
        """

        self.params = {}

        # 1계층(Input -> Hidden)
        # W1: (Input 크기, Hidden 크기) 형태
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)

        # 2계층(Hidden -> Output)
        # W2: (Hidden 크기, Output 크기) 형태
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        print("network 초기화 완료: ")
        print(f"W1 shape: {self.params['W1'].shape}")
        print(f"W2 shape: {self.params['W2'].shape}")

    def loss(self, X, y=None):
        """
        loss의 Docstring
        
        :param X: 입력 데이터(N, D)
        :param y: 정답 레이블(N,)
        """
        
        # 가중치 꺼내오기(코드 간결성)
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        N, D = X.shape

        # ------------------------------------------
        # 1. Forward Pass(순전파)
        # ------------------------------------------

        # 은닉층 계산: X * W1 + b1
        hidden_layer = X.dot(W1) + b1

        # 활성화 함수(ReLU): 0보다 작은 값을 0으로 대체
        hidden_layer_activation = np.maximum(0, hidden_layer)

        # 출력층 계산: Hidden * W2 + b2
        scores = hidden_layer_activation.dot(W2) + b2

        # 만약 정답(y)이 없으면 점수만 반환, 정답이 존재하면 Loss값을 반환.
        if y is None:
            return scores
        
        # -------------------------------------------
        # 2. Loss 계산(Softmax)
        # -------------------------------------------

        # 지수화 및 확률 변화: 값이 지나치게 커지지 않도록(안정성) 최대값을 빼기
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True)) # 이후 (N, C) 행렬과 계산을 해야 하므로
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # (N, C) 형태

        # 정답 확률에 -log 취하기
        correct_logprobs = -np.log(probs[range(N), y])
        data_loss = np.sum(correct_logprobs) / N

        # 가중치 규제 추가(L2)
        reg = 0.5
        reg_loss = 0.5 * reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
        loss = data_loss + reg_loss

        # -------------------------------------------
        # 3. Backward Pass(역전파)
        # -------------------------------------------
        grads = {}

        # dscores 계산(Softmax 미분)
        dscores = probs.copy()
        dscores[range(N), y] -= 1
        dscores /= N

        # 2계층 기울기 계산
        # W2 = Hidden을 전치 후 dscores 곱셈
        grads['W2'] = hidden_layer_activation.T.dot(dscores) + reg * W2
        grads['b2'] = np.sum(dscores, axis=0)

        # 은닉층으로 오차를 전파
        dhidden = dscores.dot(W2.T)

        # ReLU 미분(0 이하면 미분도 0)
        dhidden[hidden_layer <= 0] = 0

        # 1계층 기울기 계산
        grads['W1'] = X.T.dot(dhidden) + reg * W1
        grads['b1'] = np.sum(dhidden, axis=0)

        return loss, grads
    
    # -----------------------------------------
    #  신경망 구현
    #  ----------------------------------------
    def train(self, X, y, X_val, y_val, learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100, batch_size=200, verbose=False):
        """
        SGD를 이용한 신경망 학습
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # 손실 이력을 저장할 리스트
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            # 미니 배치 생성
            batch_indices = np.random.choice(num_train, batch_size)
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            # Loss, Gradient 계산
            loss, grads = self.loss(X_batch, y_batch)
            loss_history.append(loss)

            # 가중치 업데이트
            for param_name in self.params:
                self.params[param_name] -= learning_rate * grads[param_name]
            
            # 학습 결과 출력
            if verbose and it % 100 == 0: # varbose: 학습이 진행되는 동안 로그의 출력 여부 결정(T/F)
                print(f"Iteration {it} / {num_iters}: loss {loss:.4f}")

            # 한 Epoch(에폭)이 끝날 때마다 정확도를 계산, 학습률 감소
            if it % iterations_per_epoch == 0:
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # 학습률 감소
                learning_rate *= learning_rate_decay

        return {
            'loss_history' : loss_history,
            'train_acc_history' : train_acc_history,
            'val_acc_history' : val_acc_history,
        }
    def predict(self, X):
        """ 학습된 가중치로 예측 수행"""
        scores = self.loss(X) # 이 경우 y가 None이므로 scores만 반환
        y_pred = np.argmax(scores, axis=1)
        return y_pred

# ------------------------------------
#  cifar10 데이터 불러오기
# ------------------------------------
def get_CIFAR10_data():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # data 형변환
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # y 평탄화
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    # 서브셋 생성(검증 1000, 학습 49000)
    mask = range(49000, 50000)
    X_val = X_train[mask] # mask: 특정 데이터만 골라내기 위한 선택기
    y_val = y_train[mask]

    mask = range(49000)
    X_train = X_train[mask]
    y_train = y_train[mask]

    # 정규화: 평균 빼기
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image

    # 2차원 평탄화 (N, 32, 32, 3) -> (N, 3072)
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))

    return X_train, y_train, X_val, y_val


# -----------------------------------
# 실행 코드
# -----------------------------------
if __name__ == "__main__":
    X_train, y_train, X_val, y_val = get_CIFAR10_data()

    # hyperparameter 설정, 네트워크 생성
    input_size = 32 * 32 * 3
    hidden_size = 50 # 은닉 노드 개수. 조절하면 성능이 변화함
    num_classes = 10
    net = TwoLayerNet(input_size, hidden_size, num_classes)

    # 학습 시작
    print("\n[CIFAR-10 실전 학습 시작]")
    stats = net.train(X_train, y_train, X_val, y_val, num_iters=1500, batch_size=200,
                    learning_rate=1e-3, learning_rate_decay=0.95,
                    reg=0.25, verbose=True)

    # 정확도 확인
    val_acc = (net.predict(X_val) == y_val).mean()
    print(f'\n최종 검증 정확도(Validation Accuracy): {val_acc:.4f}')

    # 실행 결과
    # 최종 검증 정확도(Validation Accuracy): 0.4740

# ----------------------------------------------
# 가중치 시각화
# ----------------------------------------------
def visualize_grid(Xs, ubound=255.0, padding=1):
    """
    (N, H, W, C) 형태의 이미지를 그리드 형태로 시각화
    """
    (N, H, W, C) = Xs.shape # Xs: 시각화할 이미지들의 집합
    grid_size = int(np.ceil(np.sqrt(N)))
    grid_height = H * grid_size + padding * (grid_size - 1) # 이미지 사이의 간격 확보
    grid_width = W * grid_size + padding * (grid_size - 1)
    grid = np.zeros((grid_height, grid_width, C))
    next_idx = 0
    y0, y1 = 0, H
    for y in range(grid_size):
        x0, x1 = 0, W
        for x in range(grid_size):
            if next_idx < N:
                img = Xs[next_idx]
                low, high = np.min(img), np.max(img)
                grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low) # 이미지가 들어갈 좌표 범위 지정
                next_idx += 1
            x0 += W + padding
            x1 += W + padding
        y0 += H + padding
        y1 += H + padding
    return grid

# W1 가중치 가져오기 및 시각화 준비
W1 = net.params['W1']
W1 = W1.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)
# (3072, 50) -> (32, 32, 3, 50) -> (50, 32, 32, 3) 형태로 변환

# 시각화 실행
plt.imshow(visualize_grid(W1).astype('uint8'))
plt.gca().axis('off') # 그래프에서 눈금, 테두리 지우기(가시성)
plt.title('Learned weights of the First Layer (Hidden Neurons)')

# 이미지 저장
import os
save_dir = 'images/two_layer_net'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_path = f'{save_dir}/w1_visualization.png'
plt.savefig(save_path)
print(f"이미지 저장 성공: {save_path}")

# 이미지 출력
plt.show()