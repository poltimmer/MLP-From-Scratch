import pandas as pd
import math
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_csv('./HW3train.csv')

plt.scatter(df['X_0'], df['X_1'], c=df['y'], alpha=0.5)
plt.title('Training set data')
plt.xlabel('X_0')
plt.ylabel('X_1')
plt.show()

def reLu(M):
    W = np.zeros(M.shape)

    for i, row in enumerate(M):
        for j, val in enumerate(row):
            W[i, j] = max(0, val)

    return W

def softmax(raw_preds):
    """
    pass raw predictions through softmax activation function
    """
    out = np.exp(raw_preds)  # exponentiate vector of raw predictions
    # divide exponentiated vector by its sum. All values in the output sum to 1
    return out / np.sum(out)

x = df[["X_0", "X_1"]].to_numpy().T

n = x.shape[1]

W0 = np.ones((2, 10))
W1 = np.ones((10, 10))
W2 = np.ones((10, 2))

b0 = np.zeros((10, 1))
b1 = np.zeros((10, 1))
b2 = np.zeros((1, 1))

#function within reLu might be wrong
h0 = reLu(W0.T @ x + np.repeat(b0, n, axis=1))
h1 = reLu(W1.T @ h0 + np.repeat(b1, n, axis=1))
output = softmax(W2.T @ h1 + np.repeat(b2, n, axis=1))

print(output)
