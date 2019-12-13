import math

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import sklearn.neighbors
import random
import sys

def SimKNN(D):
    return sklearn.neighbors.kneighbors_graph(D, n_neighbors=10, mode="distance")

def SimEps(D):

def Lsys(W):


def spectralClustering(r, D, Sim, LaPlacian):
    W = Sim(D)
    L = LaPlacian(W)

    #(V, Lambda) = arg min norm(- L - V Lambda V)**2

    # Remove the first column
    np.delete(V, 0, 1)

    # Do k-means on V
    (X, Y) = KMeans(r).fit(V)
