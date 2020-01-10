import pandas as pd
import math
import numpy as np
from random import seed
from random import randint
from matplotlib import pyplot as plt

IN_SIZE = 2
HID_0_SIZE = 10
HID_1_SIZE = 10
OUT_SIZE = 2
learning_rate = 0.1


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

def crossEntropy(o):
    return -math.log(o, 2)


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
    h0 = reLu((W0.T @ x) + b0)
    h1 = reLu((W1.T @ h0) + b1)
    output = softmax((W2.T @ h1) + b2)

    print(output)

    # back propagation
    L = crossEntropy(output[y])

    # W0 -= learning_rate * 1 * L
    # b0 -= learning_rate * 1 * L
    #
    # W1 -= learning_rate * 1 * L
    # b1 -= learning_rate * 1 * L
    #
    # W2 -= learning_rate * 1 * L
    # b2 -= learning_rate * 1 * L



# derivatives voor alle variabelen.
# grad_w0 = x (outer prod) grad_r // geeft een matrix
# grad_b0 = grad_r
#
# grad_r = deriv_relu (elementwise mult) w1 (dot prod) * grad_p // geeft een vector
#
# grad_w1 = h_0 (outer prod) grad_p // geeft een matrix
# grad_b1 = grad_p
#
# grad_p = deriv_relu (elementwise mult) w2 (dot prod) grad_q // geeft een vector
#
# grad_w2 = h_1 (outer prod) grad_q // geeft een matrix
# grad_b2 = grad_q
#
# grad_q = deriv_sigmoid(q) * (o - y)

# een begin
def grad_q(q, o, y):
    return deriv_sigmoid(q) * (o - y)

grad_q_vec = grad_q(q, o, y)
grad_w2 = np.outer(h1, grad_q_vec)
