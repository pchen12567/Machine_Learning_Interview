import numpy as np
from sklearn.metrics import accuracy_score


class LogisticRegression(object):

    def __init__(self):
        """初始化Logistic Regression模型"""
        self.coef = None
        self.intercept = None
        self._theta = None

    def sigmoid(self, z):
        return 1. / (1. + np.exp(-z))

    def fit(self, X_train, y_train, eta=0.01, n_iters=1e4):
        """使用梯度下降法训练logistic Regression模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        # 损失函数
        def J(theta, X_b, y):
            y_hat = self.sigmoid(X_b.dot(theta))
            m = len(y)
            try:
                return -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)) / m
            except:
                return float('inf')

        # 对损失函数求偏导
        def dJ(theta, X_b, y):
            m = len(y)
            # 向量化后的公式
            return X_b.T.dot(self.sigmoid(X_b.dot(theta)) - y) / m

        # 梯度下降
        def gradient_descent(X_b, y, init_theta, alpha, n_iters=1e4, epsilon=1e-8):
            # Set init theta
            theta = init_theta

            # Set init iteration
            cur_iter = 0

            # 设置循环停止条件
            while cur_iter < n_iters:
                # 计算梯度
                gradient = dJ(theta, X_b, y)

                # 获取上一个theta
                old_theta = theta

                # 更新theta
                theta = theta - alpha * gradient

                # 设置循环停止条件
                if abs(J(theta, X_b, y) - J(old_theta, X_b, y)) < epsilon:
                    break

                # 更新循环次数
                cur_iter += 1

            # 返回梯度下降法求解出来的theta
            return theta

        # 给 X_train 增加常数列 x=1
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])

        # 设置初始化theta
        init_theta = np.zeros(X_b.shape[1])

        # 求解参数theta
        self._theta = gradient_descent(X_b, y_train, init_theta, eta, n_iters)

        # 截距
        self.intercept = self._theta[0]
        # x_i前的参数
        self.coef = self._theta[1:]

        return self

    def predict_prob(self, X_test):
        """给定待预测数据集X_test，返回表示X_test的结果概率向量"""
        assert self.intercept is not None and self.coef is not None, \
            "must fit before predict"
        assert X_test.shape[1] == len(self.coef), \
            "the feature number of X_test must be equal to X_train"

        X_b = np.hstack([np.ones((len(X_test), 1)), X_test])
        return self.sigmoid(X_b.dot(self._theta))

    def predict(self, X_test):
        """给定待预测数据集X_test，返回表示X_test的结果向量"""
        assert self.intercept is not None and self.coef is not None, \
            "must fit before predict!"
        assert X_test.shape[1] == len(self.coef), \
            "the feature number of X_test must be equal to X_train"
        prob = self.predict_prob(X_test)
        return np.array(prob >= 0.5, dtype='int')

    def score(self, X_test, y_test):
        """根据测试数据集X_test和y_test确定当前模型的准确度"""

        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)

    def __repr__(self):
        return "LogisticRegression()"
