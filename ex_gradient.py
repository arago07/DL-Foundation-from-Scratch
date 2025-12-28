import numpy as np

# f(x) = x^2
def simple_function(x):
    return x**2

# Numerical Gradient
def numerical_gradient(f, x):
    h = 1e-4 # 0.0001 <- 아주 작은 변화

    # f(x+h): h만큼 변화했을 때의 높이
    fxh = f(x + h)

    # f(x): 현재 높이
    fx = f(x)

    # 기울기: (변화한 높이) / (이동거리)
    grad = (fxh - fx) / h
    return grad

# test: x=3? 예상 결과: 6
result = numerical_gradient(simple_function, 3.0)
print(f"x=3일 때, 기울기: {result}")

# test 결과
# x=3일 때, 기울기: 6.000100000012054