import pandas as pd
import numpy as np
from kernel_svm import kernel_matrix, onevsallSVM, predict_onevsall
from scoring import get_score

data_train = pd.read_pickle('../data/histo_16s_train')
data_test = pd.read_pickle('../data/histo_16s_test')

label = pd.read_csv('../data/Ytr.csv', sep=',')

# Split the training data set : 80% training and 20% test
ind = np.random.binomial(1, 0.8, 5000)
print(np.sum(ind))

train = data.loc[ind == 1]
train_label = label.loc[ind == 1].iloc[:, 1].values

test = data.loc[ind == 0]
test_label = label.loc[ind == 0].iloc[:, 1]

# Computation of the kernel
# rho, a and b : global parameters
rho = 1.
a = 0.25
b = 1
kernel_matrix(train)

# Training
models = onevsallSVM(train, train_label, C=100)

# Predicting
ypred = []
for i in range(5000 - np.sum(ind)):
    ypred.append(onevsallSVM(models, test.iloc[i, :], train, train_label))

ypred = np.array(ypred)

# Error
print(get_score(ypred, test_label))
