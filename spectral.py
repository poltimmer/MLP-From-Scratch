import math

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import sklearn.neighbors
from sklearn import cluster, datasets, mixture
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice

# ============
# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
# ============
n_samples = 1500
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                      noise=.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
no_structure = np.random.rand(n_samples, 2), None
# Anisotropicly distributed data
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)
# blobs with varied variances
varied = datasets.make_blobs(n_samples=n_samples,
                             cluster_std=[1.0, 2.5, 0.5],
                             random_state=random_state)
datasets = [
    (noisy_circles, 2),
    (noisy_moons, 2),
    (varied, 3),
    (aniso, 3)]


def norm(x, y):
    return math.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)


def getIw(W):
    Iw = np.zeros(W.shape[0])
    for i in range(W.shape[0]):
        Iw[i] = np.sum(W[i])
    return Iw


def SimKNN(D):
    return sklearn.neighbors.kneighbors_graph(D, n_neighbors=10, mode="distance")


def SimEps(D):
    epsilon = 0.1
    n = D.shape[0]
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if norm(D[i], D[j]) < epsilon:
                W[i][j] = 1
    return W


def Lsym(W):
    IwDiag = getIw(W)
    IwDiagPowered = IwDiag ** (-1 / 2)
    Iw = np.diag(IwDiagPowered)
    return np.identity(W.shape[0]) - Iw @ W @ Iw


def spectralClustering(r, D, Sim, LaPlacian):
    W = Sim(D)
    L = LaPlacian(W)

    Lambda, V = np.linalg.eigh(L)  # TODO: select r eigenvalues/vectors with best fit for L

    # Remove the first column
    np.delete(V, 0, 1)

    # Do k-means on V
    kmeans = KMeans(r).fit(V)
    X = kmeans.cluster_centers_
    Y = kmeans.labels_
    return X, Y


Xresult, Yresult = spectralClustering(2, aniso[0], SimEps, Lsym)

LABEL_COLOR_MAP = {0: 'm',
                   1: 'g',
                   2: 'b',
                   3: 'c',
                   4: 'y',
                   }

colors = [LABEL_COLOR_MAP[l] for l in Yresult]
plt.scatter(aniso[0][:, 0], aniso[0][:, 1], s=50, c=colors)
plt.scatter(Xresult[0], Xresult[1], c='r')
plt.show()
