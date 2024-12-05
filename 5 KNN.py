import numpy as np
from collections import Counter


class KNearestNeighbors:
    def __init__(self,k=3):
        self.k= k

    def fit(self,x,y):
        self.x_train = x
        self.y_train = y

    def predict(self,x):
        predictions = [self._predict(x) for x in x]    
        return np.array(predictions)
    
    def _predict(self,x):
        distances = [self.euclidean_distance(x,x_train) for x_train in self.x_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))