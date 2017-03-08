import numpy as np
import cvxopt
from cvxopt import matrix, solvers
import pandas as pd
from numpy import diag
from numpy.linalg.linalg import matrix_rank, cholesky


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
            self.compute_kernel(X_train)
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
        n = X.shape[0]
        self.K = pd.DataFrame(np.zeros((n, n)), index=X.index, columns=X.index)
        print(self.K.shape)
        for i, row1 in X.iterrows():
            for j, row2 in X.iterrows():
                print(self.kernel_fun(row1, row2, **self.parameters))
                self.K.loc[i, j] = self.kernel_fun(row1, row2, **self.parameters)

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
        return matrix(P), matrix(q), matrix(G), matrix(h)
