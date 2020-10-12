import numpy as np
import sys

def softmax(x): #softmax 함수
    exp_a = np.exp(x - (x.max(axis=1).reshape([-1, 1])))
    exp_a /= exp_a.sum(axis=1).reshape([-1, 1])
    return exp_a

def sigmoid(z): #sigmoid 함수
    eMin = -np.log(np.finfo(type(0.1)).max)
    zSafe = np.array(np.maximum(z, eMin))
    return(1.0/(1+np.exp(-zSafe)))

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    if x.ndim == 1:
        for i in range(x.size): #bias 값일 때
            xi = x[i]
            x[i] = xi + h
            fx1 = f(x)
            x[i] = xi - h
            fx2 = f(x)
            grad[i] = (fx1 - fx2) / (2*h)
            x[i] = xi
    else:
        for i in range(x.shape[0]):
            for j in range(x.shape[1]): #weight 값일 때
                xi = x[i][j]
                x[i][j] = xi + h
                fx1 = f(x)
                x[i][j] = xi - h
                fx2 = f(x)
                grad[i][j] = (fx1 - fx2) / (2*h)
                x[i][j] = xi
    return grad