import numpy as np

class LinearRegression:
    def __init__(self, lr=0.01, epochs=1000, method='GD'):
        self.lr=lr
        self.epochs=epochs
        self.method=method
        self.weights=None
        self.bias=None

    def fit(self, X, y):
        n_samples, n_feature=X.shape

        self.weights = np.zeros(n_feature)
        self.bias = 0

        for i in range(self.epochs):
            y_hat=np.dot(X, self.weights) + self.bias
            d_dw=(1 / n_samples) * (2 * np.dot(X.T, y_hat - y))
            d_db = (1 / n_samples) * (2 * np.sum(y_hat - y))
            self.weights -= self.lr*d_dw
            self.bias -= self.lr*d_db
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias 