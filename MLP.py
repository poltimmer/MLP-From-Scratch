import pandas as pd
import math
import numpy as np
from random import seed
from random import randint
from matplotlib import pyplot as plt

IN_SIZE = 2
HID_0_SIZE = 10
HID_1_SIZE = 10
OUT_SIZE = 1
learning_rate = 0.1


df = pd.read_csv('./HW3train.csv')
'''
plt.scatter(df['X_0'], df['X_1'], c=df['y'], alpha=0.5)
plt.title('Training set data')
plt.xlabel('X_0')
plt.ylabel('X_1')
plt.show()
'''



def ReLU(v):
    w = np.zeros(v.shape)
    for i, val in enumerate(v):
        w[i] = max(0, val)

    return w


#def softmax(raw_preds):
    """
    pass raw predictions through softmax activation function
    """
#    out = np.exp(raw_preds)  # exponentiate vector of raw predictions
    # divide exponentiated vector by its sum. All values in the output sum to 1
#    return out / np.sum(out)

def ReLUDerivative(x):
    if x >= 0:
        return 1
    else:
        return 0


def sigmoid(raw_preds):
    return 1 / (1 + math.exp(-raw_preds))

def sigmoidDerivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


input = df[["X_0", "X_1"]].to_numpy().T / 10000
seed(1)

W0 = np.ones((IN_SIZE, HID_0_SIZE))
W1 = np.ones((HID_0_SIZE, HID_1_SIZE))
W2 = np.ones((HID_1_SIZE, OUT_SIZE))

b0 = np.zeros(HID_0_SIZE)
b1 = np.zeros(HID_1_SIZE)
b2 = np.zeros(OUT_SIZE)

L = 1

while (L >= 0.1):
    # take random input
    index = randint(0, input.shape[1] - 1)
    x = np.array(input[:, index]).T
    y = df.loc[index, "y"]

    # forward pass
    h0 = ReLU((W0.T @ x) + b0)
    h1 = ReLU((W1.T @ h0) + b1)
    output = sigmoid((W2.T @ h1) + b2)

    print(output)

    # back propagation
    #G(w2) = h1 * G(q)
    #G(w1) =

    #loss = crossEntropy(output[y])

    #W0 -= learning_rate * 1 * L
    #W1 -= learning_rate * 1 * L
    #W2 -= learning_rate * 1 * L

    #b0 -= learning_rate * 1 * L
    #b1 -= learning_rate * 1 * L
    #b2 -= learning_rate * 1 * L
