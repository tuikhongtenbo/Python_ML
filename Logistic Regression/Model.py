import numpy as np

class LogisticRegression:
    def __init__(self, epochs=1000, lr=0.01):
        self.epochs = epochs
        self.lr = lr
        self.weight = None
        self.bias = None

    def Sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        n_sample, n_feature = X.shape

        self.weight = np.zeros(n_feature)
        self.bias = 0

        for i in range(self.epochs):
            z = np.dot(X, self.weight) + self.bias
            y_hat = self.Sigmoid(z)
            d_dw = (1/n_sample) * np.dot(X.T, y_hat - y)
            d_db = (1/n_sample) * np.sum(y_hat - y)

            self.weight -= self.lr * d_dw
            self.bias -= self.lr * d_db

    def loss(self, X, y):
        n_sample = len(y)
        z = np.dot(X, self.weight) +  self.bias
        y_hat = self.Sigmoid(z)

        loss = -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        return loss

    def predict(self, X):
        z = np.dot(X, self.weight) + self.bias
        y_pred = self.Sigmoid(z)

        return (y_pred >= 0.5).astype(int)