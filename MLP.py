import pandas as pd
import math
import numpy as np
from matplotlib import pyplot as plt

IN_SIZE = 2
HID_0_SIZE = 10
HID_1_SIZE = 10
OUT_SIZE = 2


df = pd.read_csv('./HW3train.csv')
'''
plt.scatter(df['X_0'], df['X_1'], c=df['y'], alpha=0.5)
plt.title('Training set data')
plt.xlabel('X_0')
plt.ylabel('X_1')
plt.show()
'''



def reLu(v):
    w = np.zeros(v.shape)
    for i, val in enumerate(v):
        w[i] = max(0, val)

    return w


def softmax(raw_preds):
    """
    pass raw predictions through softmax activation function
    """
    out = np.exp(raw_preds)  # exponentiate vector of raw predictions
    # divide exponentiated vector by its sum. All values in the output sum to 1
    return out / np.sum(out)

    return result


input = df[["X_0", "X_1"]].to_numpy().T / 10000

x = np.array(input[:,0]).T
# x = input[:, 0]

W0 = np.ones((IN_SIZE, HID_0_SIZE))
W1 = np.ones((HID_0_SIZE, HID_1_SIZE))
W2 = np.ones((HID_1_SIZE, OUT_SIZE))

b0 = np.zeros(HID_0_SIZE)
b1 = np.zeros(HID_1_SIZE)
b2 = np.zeros(OUT_SIZE)

# function within reLu might be wrong
h0 = reLu((W0.T @ x) + b0)
h1 = reLu((W1.T @ h0) + b1)
output = softmax((W2.T @ h1) + b2)

print(output)
