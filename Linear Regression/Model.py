import numpy as np

class LinearRegression:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr=lr
        self.epochs=epochs
        self.weights=None

    def fit(self, X, y):
        n_samples, n_feature = X.shape

        self.weights = np.zeros((1, n_feature + 1))
        self.bias = 0
        X = np.hstack([np.ones((n_samples, 1)), X])

        self.cost = []     

        for i in range(self.epochs):
            y_hat = np.dot(X, self.weights)
            error = y_hat - y.reshape(1, -1)
            current_cost = (1 / (2 * n_samples)) * np.sum(error**2)
            d_dw = (1 / n_samples) * (np.dot(X, error))
            self.weights -= self.lr*d_dw
            self.cost.append(current_cost)
            print(self.cost[i])

    def predict(self, X):
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        return np.dot(X, self.weights) 