import math

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import sklearn.neighbors
import random
import sys

def norm(x, y):
    return math.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)

def getIw(W):
    Iw = np.zeros(W.shape[0])
    for i in range(W.shape[0]):
        Iw[i] = sum(W[i])
    return np.diag(Iw)

def SimKNN(D):
    return sklearn.neighbors.kneighbors_graph(D, n_neighbors=10, mode="distance")

def SimEps(D):
    epsilon = 0.1
    n = D.shape[0]
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if norm(D[i] - D[j]) < epsilon:
                W[i][j] = 1
    return W

def Lsym(W):
    Iw = getIw(W)
    return np.identity(W.shape[0]) - Iw**(-1/2) @ W @ Iw**(-1/2)


def spectralClustering(r, D, Sim, LaPlacian):
    W = Sim(D)
    L = LaPlacian(W)

    #(V, Lambda) = arg min norm(- L - V Lambda V)**2

    # Remove the first column
    np.delete(V, 0, 1)

    # Do k-means on V
    (X, Y) = KMeans(r).fit(V)
