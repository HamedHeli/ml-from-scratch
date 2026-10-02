"""
Implementation of k-nearest neighbours classifier
"""

import numpy as np

import utils
from utils import euclidean_dist_squared


class KNN:
    X = None
    y = None

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X  # just memorize the training data
        self.y = y

    def predict(self, X_hat):
        t = X_hat.shape[0]
        y_hat = np.zeros(t)
        D_squared = euclidean_dist_squared(X_hat, self.X)
        ## D_squared is t x n (first row is the distance of the first testing data to all training data)
        neigh_idx = D_squared.argsort(axis = 1)[:, :self.k]
        ## neigh_idx is t x k (first row is closest k points to the first testing data)
        for i in range(t):
            y_hat [i] = utils.mode(self.y[neigh_idx[i,:]])
        return(y_hat)
