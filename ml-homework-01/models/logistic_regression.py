import numpy as np

"""
Author: Wang Jianghan<ssYanhuo@foxmail.com>
Logistic Regression model using gradient descent method.
"""


def sigmoid(z):
    return 1. / (1. + np.exp(-z))


def j(theta, x, y):
    y_hat = sigmoid(x.dot(theta))
    return - np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)) / len(y)


def d_j(theta, x, y):
    y_hat = sigmoid(x.dot(theta))
    return x.T.dot(y_hat - y) / len(y)


class LogisticRegression:
    def __init__(self, learning_rate=0.0001, max_iter=10000, threshold=1e-7):
        self.theta = None
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.threshold = threshold

    def fit(self, x_train, y_train):
        x_b = np.hstack([np.ones((len(x_train), 1)), x_train])
        np.random.seed(202015005)
        self.theta = np.random.randn(x_b.shape[1])
        iter_num = 0
        while iter_num < self.max_iter:
            iter_num += 1
            last_theta = self.theta
            self.theta = self.theta - self.learning_rate * d_j(self.theta, x_b, y_train)
            if abs(j(self.theta, x_b, y_train) - j(last_theta, x_b, y_train)) < self.threshold:
                break

        return self

    def predict(self, x_predict):
        x_b = np.hstack([np.ones((len(x_predict), 1)), x_predict])
        y_predict = sigmoid(x_b.dot(self.theta))
        y_predict = np.array(y_predict >= 0.5, dtype='int')
        return y_predict
