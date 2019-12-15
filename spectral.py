import math
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import sklearn.neighbors
from sklearn import cluster, datasets, mixture
import sys
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


def fitEig(L):
    Lambda, V = np.linalg.eigh(L)
    # l_comp = V[:,40:] @ np.diag(Lambda[40:]) @ V[:,40:].T
    # eig_norm = np.linalg.norm(L-l_comp)
    # v_min = None
    # lamb_min = None
    # norm_min = sys.maxsize
    # for i in range(len(Lambda)):
    #     # print(i)
    #     for j in range(len(Lambda)):
    #         if i != j:
    #             v_new = np.array([V[:,i],V[:,j]]).T
    #             lamb_new = np.array([Lambda[i], Lambda[j]])
    #             l_comp_new = v_new @ np.diag(lamb_new) @ v_new.T
    #             norm_new = np.linalg.norm(L - l_comp_new)
    #             # v_new = V
    #             # lamb_new = Lambda
    #             # l_comp_new = L
    #             # test_speed = L - l_comp_new
    #             # speed = test_speed.T
    #             # norm_new = 1
    #             if norm_new < norm_min:
    #                 v_min = v_new
    #                 lamb_min = lamb_new
    #                 norm_min = norm_new
    # l_comp_min = v_min @ np.diag(lamb_min) @ v_min.T
    # print(norm_min)
    # print(v_min)
    # print(lamb_min)
    # return norm_min


def spectralClustering(r, D, Sim, LaPlacian):
    W = Sim(D)
    L = LaPlacian(W)

    Lambda, V = np.linalg.eigh(L)  # TODO: select r eigenvalues/vectors with best fit for L
    # fitEig(L)
    # V = V[:, -r:-1]
    # V = V[:, 1:r + 1]
    # V = V[:, -(r-1):]
    # V = V[:, 1:r]
    V = V[:, :r - 1]  # selects first r eigenvectors

    # Do k-means on V
    kmeans = KMeans(r).fit(V)
    # X = kmeans.cluster_centers_  # These cluster centers are in the laplacian space and therefore don't make sense
    # to plot
    Y = kmeans.labels_
    return Y


dataset = datasets[2]
labels = spectralClustering(dataset[1], dataset[0][0], SimEps, Lsym)

LABEL_COLOR_MAP = {0: 'm',
                   1: 'g',
                   2: 'b',
                   3: 'c',
                   4: 'y',
                   }

colors = [LABEL_COLOR_MAP[l] for l in labels]
plt.scatter(dataset[0][0][:, 0], dataset[0][0][:, 1], s=50, c=colors)
# plt.scatter(Xresult[0], Xresult[1], c='r')
plt.show()
