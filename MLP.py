import pandas as pd
import math
import numpy as np
from random import seed
from random import randint
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

IN_SIZE = 2
HID_0_SIZE = 10
HID_1_SIZE = 10
OUT_SIZE = 1
learning_rate = 0.01
RELU_LEAK = 0.01

df_train = pd.read_csv('./HW3train.csv')
df_validate = pd.read_csv('./HW3validate.csv')
loss_list = []


# plt.scatter(df['X_0'], df['X_1'], c=df['y'], alpha=0.5)
# plt.title('Training set data')
# plt.xlabel('X_0')
# plt.ylabel('X_1')
# plt.show()


def ReLU(v):  # is nu een leaky ReLU omdat ik dacht dat we te maken hadden met vanishing gradient, maar kan weg denk ik
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


def MSE(x, y):
    sum = 0
    n = len(x)
    for i in range(n):
        error = (x[i] - y[i]) ** 2
        sum += error
    return sum / n


input_training = df_train[["X_0", "X_1"]].to_numpy().T / 10000
input_validate = df_validate[["X_0", "X_1"]].to_numpy().T / 10000
y_vec_train = df_train["y"]
y_vec_validate = df_validate["y"]
seed(1)

# W0 = np.full((IN_SIZE, HID_0_SIZE), 1 / IN_SIZE)
# W1 = np.full((HID_0_SIZE, HID_1_SIZE), 1 / HID_0_SIZE)
# W2 = np.full((HID_1_SIZE, OUT_SIZE), 1 / HID_1_SIZE)

# W0 = np.random.rand(IN_SIZE, HID_0_SIZE) - 0.5
# W1 = np.random.rand(HID_0_SIZE, HID_1_SIZE) - 0.5
# W2 = np.random.rand(HID_1_SIZE, OUT_SIZE) - 0.5

# W0 = np.zeros((IN_SIZE, HID_0_SIZE))
# W1 = np.zeros((HID_0_SIZE, HID_1_SIZE))
# W2 = np.zeros((HID_1_SIZE, OUT_SIZE))

# b0 = np.random.rand(HID_0_SIZE) - 0.5
# b1 = np.random.rand(HID_1_SIZE) - 0.5
# b2 = np.random.rand(OUT_SIZE) - 0.5

# b0 = np.zeros(HID_0_SIZE)
# b1 = np.zeros(HID_1_SIZE)
# b2 = np.zeros(OUT_SIZE)

sigma = 0.9
W0 = np.random.normal(0, sigma, (IN_SIZE, HID_0_SIZE))
W1 = np.random.normal(0, sigma, (HID_0_SIZE, HID_1_SIZE))
W2 = np.random.normal(0, sigma, (HID_1_SIZE, OUT_SIZE))

b0 = np.random.normal(0, sigma, HID_0_SIZE)
b1 = np.random.normal(0, sigma, HID_1_SIZE)
b2 = np.random.normal(0, sigma, OUT_SIZE)

predicts_train = np.zeros(input_training.shape[1])
predicts_validate = np.zeros(input_validate.shape[1])
L_training = [0.5] * 1000
L_validate = 1
nrIterations = 1

while nrIterations <= 10000:
    # take random input_training
    index = randint(0, input_training.shape[1] - 1)
    x = np.array(input_training[:, index]).T
    y = df_train.loc[index, "y"]

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

    # Gradient descent
    W0 -= learning_rate * grad_W0
    b0 -= learning_rate * grad_b0

    W1 -= learning_rate * grad_W1
    b1 -= learning_rate * grad_b1

    W2 -= learning_rate * grad_W2
    b2 -= learning_rate * grad_b2

    L_training.pop(0)
    loss = (output[0] - y)**2
    L_training.append(loss)

    loss_list.append([nrIterations, sum(L_training)/len(L_training)])
    # loss_list.append([nrIterations, loss])
    nrIterations += 1

# Show output graph
# print(predicts_validate)
# plt.scatter(input_validate[0], input_validate[1], c=predicts_validate, alpha=0.5)
# plt.title('result')
# plt.xlabel('X_0')
# plt.ylabel('X_1')
# plt.show()

# Show loss function graph
loss_dataframe = pd.DataFrame(loss_list, columns=["iteration", "train"])
ax = plt.gca()

loss_dataframe.plot(kind='line',x='iteration',y='train',ax=ax)
# loss_dataframe.plot(kind='line',x='iteration',y='validate', color='red', ax=ax)

# Calculate confusion matrix for validation set
for i in range(input_validate.shape[1]):
    x = np.array(input_validate[:, i]).T

    # forward pass
    r = (W0.T @ x) + b0
    h0 = ReLU(r)
    p = (W1.T @ h0) + b1
    h1 = ReLU(p)
    q = (W2.T @ h1) + b2
    output = sigmoid(q)
    predicts_validate[i] = int(round(output[0]))

#cm = confusion_matrix(y_vec_validate, predicts_validate, labels=[0, 1])
#print(cm)
accuracy = 0
for i in range(len(predicts_validate)):
    if (y_vec_validate[i] == predicts_validate[i]):
        accuracy += 1

print("accuracy: " + str(accuracy/len(predicts_validate)))


#plt.show()

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
