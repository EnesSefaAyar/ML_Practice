# ### Import

import numpy as np
import pandas as pd


def safelog(x):
    return np.log(x + 1e-300)


# ### Importing Data

# read data into memory
data_set = np.genfromtxt("images.csv", delimiter=",")

training_data = np.array(data_set[:30000])
test_data = np.array(data_set[30000:])

data_set = np.genfromtxt("labels.csv", delimiter=",")

training_labels = np.array(data_set[:30000].astype(int))
test_labels = np.array(data_set[30000:].astype(int))

N = training_data.shape[0]
K = max(training_labels)

# ### Parameter Estimations

sample_means = np.array([np.mean(training_data[training_labels == (c + 1)], axis=0) for c in range(K)])
print("Sample Means:\n", sample_means, "\n")

sample_deviations = np.array([[0.] * training_data.shape[1]] * K)
counts = np.array([0.] * K)

for j in range(K):
    for i in range(N):
        if training_labels[i] == j + 1:
            sample_deviations[j, :] += (training_data[i, :] - sample_means[j, :]) ** 2
            counts[j] += 1
    sample_deviations[j] = sample_deviations[j] / counts[j]
    sample_deviations[j] = np.sqrt(sample_deviations[j])

print("Sample Deviations:\n", sample_deviations, "\n")

prior_probabilities = np.array([0.] * K)
for i in range(K):
    prior_probabilities[i] = counts[i] / sum(counts)

print("Prior Probabilities:\n", prior_probabilities, "\n")


# ### Parametric Classification Rule

def score_function(stddev, means, data, prior):
    global K
    N = data.shape[0]

    scores = np.array([[0.] * N] * K)

    for i in range(N):
        scores[0][i] = np.sum(safelog((1 / stddev[0] * np.sqrt(2 * np.pi))
                                      * np.exp(-(means[0] - data[i]) ** 2 / (2 * stddev[0] ** 2)))) + safelog(prior[0])
        scores[1][i] = np.sum(safelog((1 / stddev[1] * np.sqrt(2 * np.pi))
                                      * np.exp(-(means[1] - data[i]) ** 2 / (2 * stddev[1] ** 2)))) + safelog(prior[1])
        scores[2][i] = np.sum(safelog((1 / stddev[2] * np.sqrt(2 * np.pi))
                                      * np.exp(-(means[2] - data[i]) ** 2 / (2 * stddev[2] ** 2)))) + safelog(prior[2])
        scores[3][i] = np.sum(safelog((1 / stddev[3] * np.sqrt(2 * np.pi))
                                      * np.exp(-(means[3] - data[i]) ** 2 / (2 * stddev[3] ** 2)))) + safelog(prior[3])
        scores[4][i] = np.sum(safelog((1 / stddev[4] * np.sqrt(2 * np.pi))
                                      * np.exp(-(means[4] - data[i]) ** 2 / (2 * stddev[4] ** 2)))) + safelog(prior[4])
    return scores


# ### Prediction for Training Labels

scores = score_function(sample_deviations, sample_means, training_data, prior_probabilities)
y_pred = []
for i in range(training_data.shape[0]):
    maxim = max(scores[0][i], scores[1][i], scores[2][i], scores[3][i], scores[4][i])
    if maxim == scores[0][i]:
        y_pred.append(1)
    elif maxim == scores[1][i]:
        y_pred.append(2)
    elif maxim == scores[2][i]:
        y_pred.append(3)
    elif maxim == scores[3][i]:
        y_pred.append(4)
    elif maxim == scores[4][i]:
        y_pred.append(5)
y_pred = np.array(y_pred)

confusion_matrix = pd.crosstab(y_pred, training_labels, rownames=['y_predicted'], colnames=['y_train'])
print('Confusion Matrix for Training Data')
print(confusion_matrix, "\n")

# ### Prediction for Test Labels

scores = score_function(sample_deviations, sample_means, test_data, prior_probabilities)
y_test_pred = []
for i in range(test_data.shape[0]):
    maxim = max(scores[0][i], scores[1][i], scores[2][i], scores[3][i], scores[4][i])
    if maxim == scores[0][i]:
        y_test_pred.append(1)
    elif maxim == scores[1][i]:
        y_test_pred.append(2)
    elif maxim == scores[2][i]:
        y_test_pred.append(3)
    elif maxim == scores[3][i]:
        y_test_pred.append(4)
    elif maxim == scores[4][i]:
        y_test_pred.append(5)
y_test_pred = np.array(y_test_pred)

confusion_matrix = pd.crosstab(y_test_pred, test_labels, rownames=['y_predicted'], colnames=['y_train'])
print('Confusion Matrix for Test Data')
print(confusion_matrix)
