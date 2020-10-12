import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from function import *

class Two_Layer:
    def __init__(self,input_size, hidden_size, output_size):
        self.params = {}
        self.params['W1'] = np.random.randn(input_size, hidden_size) #weight1 초기값
        self.params['W2'] = np.random.randn(hidden_size, output_size) #weight2 초기값
        self.params['b1'] = np.random.randn(hidden_size) #bias1 초기값
        self.params['b2'] = np.random.randn(output_size) #bias2 초기값

    def predict(self, x):
        w1, w2 = self.params['W1'], self.params['W2'] #weight값 지정
        b1, b2 = self.params['b1'], self.params['b2'] #bias값 지정

        a1 = np.dot(x, w1) + b1 #input layer
        z1 = sigmoid(a1) #hidden layer
        a2 = np.dot(z1, w2) + b2
        y = softmax(a2) #최종 예측값에 softmax 함수를 입힌 후 반환한다. output layer
        return y

    def loss(self, x, t):
        y = self.predict(x)
        if y.ndim == 1: #one-hot encoding 일 때의 cross entropy 함수이다.
            y = y.reshape(1, y.size)
            t = t.reshape(1, t.size)
        batch_size = y.shape[0]
        cee = -np.sum(t * np.log(y+1e-7)) / batch_size
        return cee

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1) #가장 높은 인덱스 추출
        t = np.argmax(t, axis=1) #가장 높은 인덱스 추출
        cnt = 0
        for i in range(y.shape[0]):
            if y[i] == t[i]:
                cnt += 1 #y
        accuracy = (cnt / y.shape[0]) * 100
        return accuracy

    def numerical_gradient(self, x, t):
        f = lambda W:self.loss(x, t) # lambda식 f이다. 수치미분 대상 함수
        grads = {}
        grads['W1'] = numerical_gradient(f, self.params['W1']) #w1에 대한 gradient을 받아와서 dictionary에 저장한다.
        grads['b1'] = numerical_gradient(f, self.params['b1']) #b1에 대한 gradient을 받아와서 dictionary에 저장한다.
        grads['W2'] = numerical_gradient(f, self.params['W2']) #w2에 대한 gradient을 받아와서 dictionary에 저장한다.
        grads['b2'] = numerical_gradient(f, self.params['b2']) #b2에 대한 gradient을 받아와서 dictionary에 저장한다.
        return grads

    def learn(self, lr, epoch):
        iris = load_iris()

        X = iris.data  # iris data input
        y = iris.target  # iris target (label)
        y_name = iris.target_names  # iris target name

        num = np.unique(y, axis=0)
        num = num.shape[0]
        y = np.eye(num)[y]  # one-hot encoding

        num = int(X.shape[0] / 10 * 8) # 150개 랜덤하게 뒤섞은 후에 학습데이터와 테스트데이터의 비율을 8:2으로 나눔.
        select = np.random.permutation(150)  # 0~149 숫자들을 랜덤하게 뒤섞음
        Xtr, ytr = X[select[:num]], y[select[:num]]  # 120개
        Xte, yte = X[select[num:]], y[select[num:]]  # 30개

        train_size = Xtr.shape[0]
        batch_size = min(train_size, 30) #batch size가 학습할 데이터의 size보다 넘는 것을 방지한다.
        cost_y = list() #그래프를 그리기 위해 cost값들 저장
        acc_y = list() #그래프를 그리기 위해 accuracy 값들을 저장
        acc = 0.0
        acc_te = 0.0
        for i in range(epoch):
            batch_mask = np.random.choice(train_size, batch_size) #batch_size 만큼 뽑음
            x_batch = Xtr[batch_mask] #x data를 batch_size 만큼
            t_batch = ytr[batch_mask] #target data를 batch_size 만큼

            grads = self.numerical_gradient(x_batch, t_batch) #weight와 bias 값을 cost 값에 대해 편미분을 하여 반영한다.
            self.params['W1'] -= lr * grads['W1'] # weight1 값에 gradient을 반영한다.
            self.params['W2'] -= lr * grads['W2'] # weight2 값에 gradient을 반영한다.
            self.params['b1'] -= lr * grads['b1'] # bias1 값에 gradient을 반영한다.
            self.params['b2'] -= lr * grads['b2'] # bias2 값에 gradient을 반영한다.
            cost = self.loss(x_batch, t_batch) # cost 값을 얻기 위해 loss함수 작동
            acc = self.accuracy(x_batch, t_batch) # accuracy 값을 얻기 위해 accuracy 함수 작동
            acc_te = self.accuracy(Xte, yte) #test accuracy 값을 얻기 위하여 파라미터로 Xte, yte를 넣어준다.
            print("cost : ", "{:18}".format(cost), " Accuracy : ", "{:3}".format(acc))
            cost_y.append(cost) #그래프를 그리기 위하여 cost 값을 계속 넣어준다.
            acc_y.append(acc/100) #그래프를 그리기 위하여 accuracy  값을 계속 넣어준다. acc 값을 %로 표현하였기에 /100 해준 것이다.
        print("Train Accuracy : ", acc) #Train Accuracy 마지막 값.
        print("Test Accuracy : ", acc_te) #Test Accuracy값
        cost_x = np.arange(0, epoch, 1) #cost 그래프 x축
        acc_x = np.arange(0, epoch, 1) #accuracy 그래프 x축
        cost_y = np.array(cost_y) #cost 그래프 y축
        acc_y = np.array(acc_y) #accuaracy 그래프 y축
        plt.plot(cost_x, cost_y, label="cost")  # cost plot
        plt.plot(acc_x, acc_y, label="accuracy")  # accuracy plot
        plt.xlabel("number of iterations")
        plt.ylabel("cost")
        plt.legend()
        plt.show()



