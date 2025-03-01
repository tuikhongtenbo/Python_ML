import numpy as np

class LinearRegression:
    def __init__(self, lr=0.001, epochs=10000):
        self.lr=lr
        self.epochs=epochs
        self.weights=None
        self.cost = []  

    def fit(self, X, y):
        n_samples, n_feature = X.shape

        self.weights = np.ones((n_feature + 1, 1))
        X = np.hstack([np.ones((n_samples, 1)), X])
        y = y.reshape(-1, 1)   

        for i in range(self.epochs):
            y_hat = np.dot(X, self.weights)
            error = y_hat - y
            current_cost = (1 / (2 * n_samples)) * np.sum(error**2)
            d_dw = (1 / n_samples) * (np.dot(X.T, error))
            self.weights -= self.lr*d_dw
            self.cost.append(current_cost)
            if i % 100 == 0:
                print(f'Epoch: {i}, Loss: {current_cost}')

    def predict(self, X):
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        return np.dot(X, self.weights) 