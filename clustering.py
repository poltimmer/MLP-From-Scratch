import math

import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics.cluster import normalized_mutual_info_score
import random
import sys

# dataset 1
Data, y = make_blobs(n_samples=15000, centers=5, cluster_std=[3.9, 1.7, 1.5, 5.9, 2.8], n_features=2, random_state=10,
                     center_box=(-35.0, 25.0))
Data = np.vstack(
    (Data[y == 0][:5000], Data[y == 1][:4500], Data[y == 2][:4000], Data[y == 3][:2000], Data[y == 4][:1000]))

y = np.hstack((y[y == 0][:5000], y[y == 1][:4500], y[y == 2][:4000], y[y == 3][:2000], y[y == 4][:1000]))

# dataset 2
X2, y2 = make_blobs(n_samples=3500, cluster_std=[1.0, 2.5, 0.5], random_state=170, center_box=(-15.0, 5.0))


def euclidian_dist(x, y):
    return math.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)


def manhattan_dist(x, y):
    return abs(x[0] - y[0]) + abs(x[1] - y[1])


def init_centroids(D, r, init, dist):
    X = np.zeros((D.shape[1], r))
    if init == "random":
        for x in range(X.shape[0]):
            for y in range(X.shape[1]):
                X[x, y] = random.randrange(-15, 5.0)

    if init == "forgy":
        dataset = [row.T for row in D]
        for y in range(X.shape[1]):
            X[:, y] = random.sample(dataset, 1)[0]

    if init == "k-means++":
        s = 1
        r = X.shape[1]
        dataset = [row.T for row in D]
        X_new = np.zeros((D.shape[1], 1))
        X_new[:, 0] = random.sample(dataset, 1)[0]
        X = X_new
        while s < r:
            s += 1
            dists = []
            for row in D:
                min_dist = sys.maxsize
                for i in range(X.shape[1]):
                    distance = dist(row, X[:,i])
                    if distance < min_dist:
                        min_dist = distance
                dists.append(min_dist)
            dists_squared = [dist**2 for dist in dists]
            sum_squared = np.sum(dists_squared)
            probabilities = [dist/sum_squared for dist in dists_squared]
            choice = np.random.choice(range(D.shape[0]), p=probabilities)
            X_new = np.zeros((D.shape[1], 1))
            X_new[:, 0] = D[choice, :]
            X = np.c_[X, X_new]

    return X


def cluster_assignments(X, D, dist):  ## TODO: check if rows/columns are queried correctly
    Y = np.zeros((D.shape[0], X.shape[1]))
    for i in range(D.shape[0]):  # For each datapoint
        min_dist = sys.maxsize
        for t in range(X.shape[1]):  # For each centroid
            if dist(D[i], X[:, t]) < min_dist:
                min_dist = dist(D[i], X[:, t])
                Y[i, :] = np.zeros(len(Y[i]))
                Y[i, t] = 1
    return Y


def centroid_update(Y, D):
    X = np.zeros((D.shape[1], Y.shape[1]))
    for s in range(X.shape[1]):
        sum = np.zeros(X.shape[0])
        for i in range(Y.shape[0]):
            if Y[i, s] == 1:
                sum = sum + D[i, :].T

        X[:, s] = (1 / np.linalg.norm(Y[:, s], ord=1)) * sum
    return X


def k_means(r, D, init, dist):
    X = init_centroids(D, r, init, dist)
    for i in range(10):
        print(i)
        Y = cluster_assignments(X, D, dist)
        X = centroid_update(Y, D)
    return X, Y

sum = 0

for i in range(5):
    X, Y = k_means(3, X2, "random", manhattan_dist)
    clusters = []
    for row in range(Y.shape[0]):
        for col in range(Y.shape[1]):
            if Y[row][col] == 1:
                clusters.append(col)

    LABEL_COLOR_MAP = {0: 'm',
                       1: 'g',
                       2: 'b',
                       3: 'c',
                       4: 'y',
                       }

    colors = [LABEL_COLOR_MAP[l] for l in clusters]

    plt.scatter(X2[:, 0], X2[:, 1], s=50, c=colors)
    plt.scatter(X[0], X[1], c='r')
    plt.show()

    sum += normalized_mutual_info_score(y2, clusters)

average = sum/5
print(average)
