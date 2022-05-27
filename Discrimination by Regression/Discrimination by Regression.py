import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ### Parameters

means = np.array([[0, 2.5],
                  [-2.5, -2],
                  [2.5, -2]])

varmat1 = np.array([[3.2, 0],
                    [0, 1.2]])

varmat2 = np.array([[1.2, 0.8],
                    [0.8, 1.2]])

varmat3 = np.array([[1.2, -0.8],
                    [-0.8, 1.2]])

class_sizes = np.array([120, 80, 100])

N = sum(class_sizes)
K = class_sizes.shape[0]

eta = 0.01
epsilon = 0.001

# ### Random Data Generation

np.random.seed(521)

points1 = np.random.multivariate_normal(means[0], varmat1, class_sizes[0])
points2 = np.random.multivariate_normal(means[1], varmat2, class_sizes[1])
points3 = np.random.multivariate_normal(means[2], varmat3, class_sizes[2])
points = np.vstack((points1, points2, points3))

labels = np.concatenate([[1] * class_sizes[0], [2] * class_sizes[1], [3] * class_sizes[2]])

# Rearrange labels for descendant loops
y_truth = np.zeros((N, K)).astype(int)
y_truth[range(N), labels - 1] = 1

# ### Plotting Data

plt.figure(figsize=(10, 6))
plt.plot(points1[:, 0], points1[:, 1], "r.", markersize=10)
plt.plot(points2[:, 0], points2[:, 1], "g.", markersize=10)
plt.plot(points3[:, 0], points3[:, 1], "b.", markersize=10)
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()


# Define the sigmoid function
def sigmoid(points, w, w0):
    return 1 / (1 + np.exp(-(np.matmul(points, w) + w0)))


# Define gradient functions
def gradient_w(points, y_truth, y_predicted):
    return -np.matmul(points.T, ((y_truth - y_predicted) * y_predicted * (1 - y_predicted)))


def gradient_w0(y_truth, y_predicted):
    return -np.sum((y_truth - y_predicted) * y_predicted * (1 - y_predicted), axis=0)


# Initialize parameters
np.random.seed(521)
w = np.random.uniform(low=-0.01, high=0.01, size=(points.shape[1], K))
w0 = np.random.uniform(low=-0.01, high=0.01, size=(1, K))

# learn w and w0 using gradient descent
iteration = 1
objective_values = []
while 1:
    y_predicted = sigmoid(points, w, w0)

    objective_values = np.append(objective_values, 0.5 * np.sum((y_truth - y_predicted) ** 2))

    w_old = w
    w0_old = w0
    w = w - eta * gradient_w(points, y_truth, y_predicted)
    w0 = w0 - eta * gradient_w0(y_truth, y_predicted)

    if np.sqrt(np.sum((w0 - w0_old)) ** 2 + np.sum((w - w_old) ** 2)) < epsilon:
        break

    iteration = iteration + 1

print(w)
print(w0)

# Plot objective function calculated during iterations
plt.figure(figsize=(10, 6))
plt.plot(range(1, iteration + 1), objective_values, "k-")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.show()

# Calculate confusion matrix

predictions = np.argmax(y_predicted, axis=1) + 1
confusion_matrix = pd.crosstab(predictions, labels, rownames=['y_pred'], colnames=['y_truth'])
print(confusion_matrix)

# Plot boundaries and indicate misclassified data points

x1_interval = np.linspace(-6, +6, 100 * int(np.amax(points)))
x2_interval = np.linspace(-6, +6, 100 * int(np.amax(points)))
x1_grid, x2_grid = np.meshgrid(x1_interval, x2_interval)

discriminant_values = np.zeros((len(x1_interval), len(x2_interval), K))

for i in range(K):
    for j in range(x1_grid.shape[0]):
        for k in range(x2_grid.shape[0]):
            x = np.array([x1_grid[j][k], x2_grid[j][k]])
            discriminant_values[j][k] = (1 / (1 + np.exp(-(np.matmul(x, w) + w0))))

A = discriminant_values[:, :, 0]
B = discriminant_values[:, :, 1]
C = discriminant_values[:, :, 2]

A[(A < B) & (A < C)] = np.nan
B[(B < A) & (B < C)] = np.nan
C[(C < A) & (C < B)] = np.nan

plt.figure(figsize=(6, 6))
plt.plot(points[labels == 1, 0], points[labels == 1, 1], "r.", markersize=10)
plt.plot(points[labels == 2, 0], points[labels == 2, 1], "g.", markersize=10)
plt.plot(points[labels == 3, 0], points[labels == 3, 1], "b.", markersize=10)

plt.contour(x1_grid, x2_grid, B - C, levels=0, colors="k")
plt.contour(x1_grid, x2_grid, A - B, levels=0, colors="k")
plt.contour(x1_grid, x2_grid, A - C, levels=0, colors="k")

plt.plot(points[predictions != labels, 0], points[predictions != labels, 1], "ko", markersize=12, fillstyle="none")

plt.ylabel("x2")
plt.xlabel("x1")
plt.show()
