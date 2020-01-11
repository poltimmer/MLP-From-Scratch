import pandas as pd
import math
import numpy as np
from random import seed
from random import randint
from matplotlib import pyplot as plt

IN_SIZE = 2
HID_0_SIZE = 10
HID_1_SIZE = 5
OUT_SIZE = 2
learning_rate = 0.01

df = pd.read_csv('./HW3train.csv')

plt.scatter(df['X_0'], df['X_1'], c=df['y'], alpha=0.5)
plt.title('Training set data')
plt.xlabel('X_0')
plt.ylabel('X_1')
plt.show()



def ReLU(v):
    w = np.zeros(v.shape)
    for i, val in enumerate(v):
        w[i] = max(0, val)

    return w

    # def softmax(raw_preds):
    """
    pass raw predictions through softmax activation function
    """


#    out = np.exp(raw_preds)  # exponentiate vector of raw predictions
# divide exponentiated vector by its sum. All values in the output sum to 1
#    return out / np.sum(out)

def ReLUDerivative(x):
    for n in np.nditer(x, op_flags=['readwrite']):
        n[...] = 1 if n >= 0 else 0

    return x


def sigmoid(raw_preds):
    for i, gamma in enumerate(raw_preds):
        if gamma < 0:
            raw_preds[i] = 1 - 1 / (1 + math.exp(gamma))
        else:
            raw_preds[i] = 1 / (1 + math.exp(-gamma))

    return raw_preds


def sigmoidDerivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def grad_o(o, y):
    return o - np.array([y, 1 - y])


def grad_q(q, o, y):
    return sigmoidDerivative(q) * grad_o(o, y)


def grad_p(x, W2, g_q):
    return np.multiply(ReLUDerivative(x), W2 @ g_q)


def grad_r(x, W1, g_p):
    return np.multiply(ReLUDerivative(x), W1 @ g_p)


input = df[["X_0", "X_1"]].to_numpy().T / 10000
seed(1)

# W0 = np.full((IN_SIZE, HID_0_SIZE), 1 / IN_SIZE)
# W1 = np.full((HID_0_SIZE, HID_1_SIZE), 1 / HID_0_SIZE)
# W2 = np.full((HID_1_SIZE, OUT_SIZE), 1 / HID_1_SIZE)

W0 = np.ones((IN_SIZE, HID_0_SIZE))
W1 = np.ones((HID_0_SIZE, HID_1_SIZE))
W2 = np.ones((HID_1_SIZE, OUT_SIZE))

b0 = np.zeros(HID_0_SIZE)
b1 = np.zeros(HID_1_SIZE)
b2 = np.zeros(OUT_SIZE)

L = 1

while L >= 0.1:
    # take random input
    index = randint(0, input.shape[1] - 1)
    x = np.array(input[:, index]).T
    y = df.loc[index, "y"]

    # forward pass
    r = (W0.T @ x) + b0
    h0 = ReLU(r)
    p = (W1.T @ h0) + b1
    h1 = ReLU(p)
    q = (W2.T @ h1) + b2
    output = sigmoid(q)

    # back propagation
    grad_q_vec = grad_q(q, output, y)
    grad_W2 = np.outer(h1, grad_q_vec)
    grad_b2 = grad_q_vec

    grad_p_vec = grad_p(p, W2, grad_q_vec)

    grad_W1 = np.outer(h0, grad_p_vec)
    grad_b1 = grad_p_vec

    grad_r_vec = grad_r(r, W1, grad_p_vec)

    grad_W0 = np.outer(x, grad_r_vec)
    grad_b0 = grad_r_vec

    W0 -= learning_rate * grad_W0
    b0 -= learning_rate * grad_b0

    W1 -= learning_rate * grad_W1
    b1 -= learning_rate * grad_b1

    W2 -= learning_rate * grad_W2
    b2 -= learning_rate * grad_b2

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
