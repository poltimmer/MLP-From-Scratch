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
RELU_LEAK = 0.01

df = pd.read_csv('./HW3train.csv')

# plt.scatter(df['X_0'], df['X_1'], c=df['y'], alpha=0.5)
# plt.title('Training set data')
# plt.xlabel('X_0')
# plt.ylabel('X_1')
# plt.show()


def ReLU(v): # is nu een leaky ReLU omdat ik dacht dat we te maken hadden met vanishing gradient, maar kan weg denk ik
    w = np.zeros(v.shape)
    for i, val in enumerate(v):
        w[i] = max(RELU_LEAK * val, val)

    return w

def ReLUDerivative(v):
    res = np.copy(v)
    for n in np.nditer(res, op_flags=['readwrite']):
        n[...] = 1 if n >= 0 else RELU_LEAK

    return res


def sigmoid(raw_preds):
    res = np.zeros(raw_preds.shape)
    for i, gamma in enumerate(raw_preds):
        if gamma < 0:
            res[i] = 1 - 1 / (1 + math.exp(gamma))
        else:
            res[i] = 1 / (1 + math.exp(-gamma))

    return res


def sigmoidDerivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def grad_o(o, y):
    # return o - np.array([y, 1 - y])
    return o - y


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

W0 = np.random.rand(IN_SIZE, HID_0_SIZE) - 0.5
W1 = np.random.rand(HID_0_SIZE, HID_1_SIZE) - 0.5
W2 = np.random.rand(HID_1_SIZE, OUT_SIZE) - 0.5

# W0 = np.ones((IN_SIZE, HID_0_SIZE))
# W1 = np.ones((HID_0_SIZE, HID_1_SIZE))
# W2 = np.ones((HID_1_SIZE, OUT_SIZE))

b0 = np.random.rand(HID_0_SIZE) - 0.5
b1 = np.random.rand(HID_1_SIZE) - 0.5
b2 = np.random.rand(OUT_SIZE) - 0.5

# b0 = np.zeros(HID_0_SIZE)
# b1 = np.zeros(HID_1_SIZE)
# b2 = np.zeros(OUT_SIZE)

L = 1 # wordt nog niet gebruikt

for i in range(25000):
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


predicts = np.zeros(input.shape[1])
for i in range(input.shape[1]):
    x = np.array(input[:, i]).T
    y = df.loc[i, "y"]

    # forward pass
    r = (W0.T @ x) + b0
    h0 = ReLU(r)
    p = (W1.T @ h0) + b1
    h1 = ReLU(p)
    q = (W2.T @ h1) + b2
    output = sigmoid(q)
    predicts[i] = int(round(output[0]))

print(predicts)
plt.scatter(input[0], input[1], c=predicts, alpha=0.5)
plt.title('result')
plt.xlabel('X_0')
plt.ylabel('X_1')
plt.show()


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
