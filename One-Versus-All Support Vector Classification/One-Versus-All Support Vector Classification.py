import cvxopt as cvx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.spatial.distance as dt

# read data into memory
data_set = np.genfromtxt("images.csv", delimiter=",")

# Seperate training and test data.
x_train = data_set[:1000]
x_test = data_set[1000:]

# read data into memory
data_set = np.genfromtxt("labels.csv", delimiter=",")

# Seperate labels of training and test data sets
y_train = data_set[:1000]
y_test = data_set[1000:]


# ## Distance and Kernel Functions

# define Gaussian kernel function
def gaussian_kernel(X1, X2, s):
    D = dt.cdist(X1, X2)
    K = np.exp(-D ** 2 / (2 * s ** 2))
    return (K)


# ## Learning Algorithm
#
def train(x_test, x_train, y_train, C=10):
    # calculate Gaussian kernel
    s = 10
    K_test = gaussian_kernel(x_test, x_train, s)
    K_train = gaussian_kernel(x_train, x_train, s)
    yyK = np.matmul(y_train[:, None], y_train[None, :]) * K_train

    # set learning parameters
    epsilon = 1e-3

    P = cvx.matrix(yyK)
    q = cvx.matrix(-np.ones((len(x_train), 1)))
    G = cvx.matrix(np.vstack((-np.eye(len(x_train)), np.eye(len(x_train)))))
    h = cvx.matrix(np.vstack((np.zeros((len(x_train), 1)), C * np.ones((len(x_train), 1)))))
    A = cvx.matrix(1.0 * y_train[None, :])
    b = cvx.matrix(0.0)

    # use cvxopt library to solve QP problems
    result = cvx.solvers.qp(P, q, G, h, A, b)
    alpha = np.reshape(result["x"], len(x_train))
    alpha[alpha < C * epsilon] = 0
    alpha[alpha > C * (1 - epsilon)] = C

    # find bias parameter
    support_indices, = np.where(alpha != 0)
    active_indices, = np.where(np.logical_and(alpha != 0, alpha < C))
    w0 = np.mean(
        y_train[active_indices] * (1 - np.matmul(yyK[np.ix_(active_indices, support_indices)], alpha[support_indices])))

    # calculate predictions on training samples
    f_predicted = np.matmul(K_train, y_train[:, None] * alpha[:, None]) + w0
    t_predicted = np.matmul(K_test, y_train[:, None] * alpha[:, None]) + w0

    return f_predicted, t_predicted


y = np.array([-np.ones(len(y_train))] * 5)
for i in range(len(y_train)):
    if y_train[i] == 1:
        y[0][i] = 1
    elif y_train[i] == 2:
        y[1][i] = 1
    elif y_train[i] == 3:
        y[2][i] = 1
    elif y_train[i] == 4:
        y[3][i] = 1
    elif y_train[i] == 5:
        y[4][i] = 1

f1, t1 = train(x_test, x_train, y[0])
f2, t2 = train(x_test, x_train, y[1])
f3, t3 = train(x_test, x_train, y[2])
f4, t4 = train(x_test, x_train, y[3])
f5, t5 = train(x_test, x_train, y[4])

f = np.hstack((f1, f2, f3, f4, f5))
t = np.hstack((t1, t2, t3, t4, t5))

# ## Training Performance

y_pred_f = np.argmax(f, axis=1) + 1

confusion_matrix = pd.crosstab(np.reshape(y_pred_f, len(y_train)), y_train, rownames=['y_predicted'],
                               colnames=['y_train'])
print(confusion_matrix)

y_pred_t = np.argmax(t, axis=1) + 1

confusion_matrix2 = pd.crosstab(np.reshape(y_pred_t, len(y_test)), y_test, rownames=['y_predicted'],
                                colnames=['y_test'])
print(confusion_matrix2)


# # Accuracy


def accuracy(prediction, actual):
    correct = 0
    for i, guess in enumerate(prediction):
        if guess == actual[i]:
            correct += 1

    return correct / len(actual)


acc_test = np.array([0.] * 5)
acc_train = np.array([0.] * 5)
C = [0.1, 1, 10, 100, 1000]
for j, i in enumerate(C):
    f1, t1 = train(x_test, x_train, y[0], i)
    f2, t2 = train(x_test, x_train, y[1], i)
    f3, t3 = train(x_test, x_train, y[2], i)
    f4, t4 = train(x_test, x_train, y[3], i)
    f5, t5 = train(x_test, x_train, y[4], i)

    f = np.hstack((f1, f2, f3, f4, f5))
    t = np.hstack((t1, t2, t3, t4, t5))

    y_pred_f = np.argmax(f, axis=1) + 1
    y_pred_t = np.argmax(t, axis=1) + 1

    acc_test[j] = accuracy(y_pred_t, y_test)
    acc_train[j] = accuracy(y_pred_f, y_train)

fig = plt.figure(figsize=(10, 6))
plt.plot(C, acc_train, color="blue", label="training")
plt.scatter(C, acc_train, color="blue")
plt.plot(C, acc_test, color="red", label="test")
plt.scatter(C, acc_test, color="red")
plt.xlabel("Regularization Parameter")
plt.ylabel("Accuracy")
plt.legend(loc="upper left")
plt.xscale("log")
plt.show()
