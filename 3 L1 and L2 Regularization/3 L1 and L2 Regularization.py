import numpy as np

class LinearRegression:

    def __init__(self, learning_rate=0.001, iterations=1000, l1_reg=0.0, l2_reg=0.0):
        self.lr = learning_rate
        self.n_iters = iterations
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias

            
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

           
            dw += self.l1_reg * np.sign(self.weights) + 2 * self.l2_reg * self.weights

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred
