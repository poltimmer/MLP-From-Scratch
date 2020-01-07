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

W1 = np.random.rand(2, 10)
W2 = np.random.rand(10, 10)
W3 = np.random.rand(10, 1)

b1 = np.random.rand()

def reLu(M):
    W = np.zeros(M.shape)

    for i, row in enumerate(M):
        for j, val in enumerate(row):
            W[i, j] = max(0, val)

    return W

D = reLu(L)

def softmax(raw_preds):
    """
    pass raw predictions through softmax activation function
    """
    out = np.exp(raw_preds)  # exponentiate vector of raw predictions
    # divide exponentiated vector by its sum. All values in the output sum to 1
    return out / np.sum(out)

x = df[["X_0", "X_1"]].to_numpy()

h0 = reLu(x)
h1 = reLu(h0)
output = softmax(h1)

print(output)
