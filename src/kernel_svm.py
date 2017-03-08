import numpy as np
import cvxopt
from cvxopt import matrix, solvers
import pandas as pd
from numpy import diag

rho = 1.
a = 0.25
b = 1



class KernelSVM:
    def __init__(self, K=None, kernel_fun=None, C=300, **kwargs):
        self.given_kernel = False
        if K is None and kernel_fun is None:
            print("Give K or kernel_fun")

        if K is not None:
            self.given_kernel = True
        self.kernel_fun = kernel_fun
        self.parameters = kwargs
        self.C = C
        self.K = K
        self.sol = None
        self.alpha_y = None
        self.support_ = None
        self.support_vectors_ = None
        self.alphas = None

    def fit(self, X_train, y_train):
        # Need X_test to compute the kernel
        if not self.given_kernel:
            self.K = self.compute_kernel(X_train)
        Q, p, G, h = self.transform_dual(y_train)
        self.sol = solvers.qp(Q, p, G, h)
        self.alphas = pd.Series(self.sol['x'], index=X_train.index)
        self.alpha_y = pd.Series(np.multiply(self.alphas, (np.array(y_train))), index=X_train.index)

        self.support_ = self.alpha_y[(10 ** (-6) < self.alpha_y) & (self.alpha_y < self.C - 10 ** (-6))].index
        self.support_vectors_ = X_train.loc[self.support_]
        self.alphas = pd.Series(self.sol['x'], index=X_train.index)

    def predict(self, X_test):
        """Return predictions for test data"""
        return self.predict_margin(X_test).apply(np.sign)

    def predict_margin(self, X_test):
        """Return prediction margins for test data"""
        return X_test.apply(self.predict_x, axis=1)

    def predict_x(self, x):
        """Return predictions for a vector"""
        if self.given_kernel:
            return self.alphas.dot(self.K[self.support_, x])
        return self.alphas.dot(
            pd.Series([self.kernel_fun(support_vec, x) for _, support_vec in self.support_vectors_.iterrows()],
                      index=self.support_))

    def compute_kernel(self, X):
        data = X.values
        n = X.shape[0]
        K = np.zeros((n, n))
        for i in range(n):
            if (i % 500 == 0):
                print(i)
            for j in range(n):
                K[i, j] = self.kernel_fun(data[i, :], data[j, :], **self.parameters)
        return K

    def transform_dual(self, y):
        # Here, we keep lambda as the variable. We need to make A bigger to account for the 2 inequalities on lambda
        n = self.K.shape[0]
        P = self.K
        q = - y * np.ones(len(y))
        G = np.zeros((2 * n, n))
        h = np.zeros(2 * n)
        G[:n, :] = matrix(np.diag(y))
        G[n:2 * n, :] = matrix(-np.diag(y))
        h[:n] = self.C * np.ones(n)
        print(matrix(q).size)
        return matrix(P), matrix(q), matrix(G), matrix(h)





def onevsallSVM(X, y, C=300):
    models = {}
    for cls in np.unique(y):
        X_train = X.copy()
        y_train = pd.Series([-1 if yval == cls else 1 for yval in y])
        models[cls] = svm(X_train, y_train, C=C)
    return models


def predict_onevsall(models, x, X):
    cls, margin = 0, 0
    ker = np.array([kernel(x, X[i]) for i in range(X.shape[0])])
    for i, m in models.items():
        a = np.dot(m, ker)
        if np.abs(a) > margin and a > 0:
            cls, margin = i, np.abs(a)
    return cls
