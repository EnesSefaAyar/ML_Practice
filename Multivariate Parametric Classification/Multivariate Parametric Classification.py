# ### Enes Sefa Ayar

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ## Parameters

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

# ## Data Generation

np.random.seed(521)

points1 = np.random.multivariate_normal(means[0], varmat1, class_sizes[0])
points2 = np.random.multivariate_normal(means[1], varmat2, class_sizes[1])
points3 = np.random.multivariate_normal(means[2], varmat3, class_sizes[2])
points = np.vstack((points1, points2, points3))

labels = np.concatenate([[1] * class_sizes[0], [2] * class_sizes[1], [3] * class_sizes[2]])

# ## Plotting Data

plt.figure(figsize=(6, 6))
plt.plot(points1[:, 0], points1[:, 1], "r.", markersize=10)
plt.plot(points2[:, 0], points2[:, 1], "g.", markersize=10)
plt.plot(points3[:, 0], points3[:, 1], "b.", markersize=10)
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

# ## Parameter Estimation

mean1 = np.array([np.mean(points1[:, 0]), np.mean(points1[:, 1])])
mean2 = np.array([np.mean(points2[:, 0]), np.mean(points2[:, 1])])
mean3 = np.array([np.mean(points3[:, 0]), np.mean(points3[:, 1])])
sample_means = np.array([mean1, mean2, mean3])
print("Sample Means:\n", sample_means, "\n")


def find_covariance(points, mean):
    covariance = [[0.0, 0.0], [0.0, 0.0]]
    for i in range(len(points[:, 0])):
        covariance += np.matmul(np.array([points[i] - mean]).T, np.array([points[i] - mean]))
    return covariance / len(points[:, 0])


sample_covariences = np.array(
    [find_covariance(points1, mean1), find_covariance(points2, mean2), find_covariance(points3, mean3)])

print("Sample Variances:\n", sample_covariences, "\n")

class_priors = [class_sizes[i] / np.sum(class_sizes) for i in range(K)]
print("Class Priors:\n", class_priors, "\n")

# ## Score Functions

det_covars = [np.linalg.det(sample_covariences[i]) for i in range(K)]
inv_covars = [np.linalg.inv(sample_covariences[i]) for i in range(K)]

Wc = [-0.5 * inv_covars[i] for i in range(K)]
wc = [np.matmul(inv_covars[i], sample_means[i]) for i in range(K)]
wc0 = [-0.5 * np.matmul(np.array(sample_means[i]).T, wc[i]) - np.log(2 * math.pi) - 0.5 * np.log(det_covars[i])
       + np.log(class_priors[i]) for i in range(K)]

score1 = [
    np.matmul(np.matmul(np.array(points[i]).T, Wc[0]), points[i]) + np.matmul(np.array(wc[0]).T, points[i]) + wc0[0]
    for i in range(N)]
score2 = [
    np.matmul(np.matmul(np.array(points[i]).T, Wc[1]), points[i]) + np.matmul(np.array(wc[1]).T, points[i]) + wc0[1]
    for i in range(N)]
score3 = [
    np.matmul(np.matmul(np.array(points[i]).T, Wc[2]), points[i]) + np.matmul(np.array(wc[2]).T, points[i]) + wc0[2]
    for i in range(N)]

# ## Confusion Matrix

predictions = np.zeros(N, dtype=int)
for i in range(N):
    maximum = np.amax(np.array([score1[i], score2[i], score3[i]]))
    if score1[i] == maximum:
        predictions[i] = 1
    elif score2[i] == maximum:
        predictions[i] = 2
    else:
        predictions[i] = 3
confusion_matrix = pd.crosstab(predictions, labels, rownames=['y_pred'], colnames=['y_truth'])

print(confusion_matrix)

# ## Decision Boundaries

x1_interval = np.linspace(-6, +6, 100 * int(np.amax(points)))
x2_interval = np.linspace(-6, +6, 100 * int(np.amax(points)))
x1_grid, x2_grid = np.meshgrid(x1_interval, x2_interval)

discriminant_values = np.zeros((len(x1_interval), len(x2_interval), K))
for i in range(K):
    for j in range(x1_grid.shape[0]):
        for k in range(x2_grid.shape[0]):
            x = np.array([x1_grid[j][k], x2_grid[j][k]]).reshape(2, 1)
            discriminant_values[j, k, i] = np.matmul(np.matmul(np.array(x).T, Wc[i]), x) + np.matmul(np.array(wc[i]).T,
                                                                                                     x) + wc0[i]

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
