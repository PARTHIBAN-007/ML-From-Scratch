import numpy as np
class LinearRegressionScratch:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.beta_0 = 0 
        self.beta_1 = 0  

    def predict(self, X):
        return self.beta_0 + self.beta_1 * X

    def cost_function(self, X, y):
        n = len(y)
        y_pred = self.predict(X)
        return (1 / n) * np.sum((y - y_pred) ** 2)

    def fit(self, X, y):
        n = len(y)

        for i in range(self.iterations):
            y_pred = self.predict(X)
            

            d_beta_0 = (-2 / n) * np.sum(y - y_pred)
            d_beta_1 = (-2 / n) * np.sum((y - y_pred) * X)

            self.beta_0 -= self.learning_rate * d_beta_0
            self.beta_1 -= self.learning_rate * d_beta_1

            if i % 100 == 0:
                cost = self.cost_function(X, y)
                print(f"Iteration {i}: Cost {cost:.4f}")


X = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([1, 2, 3, 4, 5], dtype=float)

model = LinearRegressionScratch()

model.fit(X, y)


print(f"Intercept (beta_0) : {model.beta_0}")
print(f"Slope (beta_1) : {model.beta_1}")


