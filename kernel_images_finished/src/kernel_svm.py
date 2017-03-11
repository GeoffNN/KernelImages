import numpy as np
import cvxopt
from cvxopt import matrix, solvers
import pandas as pd
from numpy import diag


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
        self.K_train_index = None

    def fit(self, X_train, y_train, K_train_index = None):
        # Need X_test to compute the kernel
        self.K_train_index = K_train_index
        if not self.given_kernel:
            self.compute_kernel(X_train)
            self.K_train_index = range(X_train.shape[0])
        Q, p, G, h = self.transform_dual(y_train)

        self.sol = solvers.qp(Q, p, G, h)
        self.alphas = pd.Series(self.sol['x'], index=X_train.index)
        self.alpha_y = pd.Series(np.multiply(self.alphas, (np.array(y_train))), index=X_train.index)
        
        self.support_ = self.alpha_y[(10 ** (-3) < self.alpha_y) & (self.alpha_y < self.C - 10 ** (-3))].index
        self.support_vectors_ = X_train.loc[self.support_]
        
        self.alphas = self.alphas.loc[self.support_]
        self.alpha_y = self.alphas.loc[self.support_]

        
    def predict(self, X_test, X_test_index = None):
        """Return predictions for test data"""
        return self.predict_margin(X_test, X_test_index).apply(np.sign)

    def predict_margin(self, X_test, X_test_index = None):
        """Return prediction margins for test data"""
        if self.given_kernel:
            return(pd.Series([self.alphas.dot(
            [self.K[support_index, x_index] for support_index in self.support_]) for x_index in X_test_index]))
            
        return X_test.apply(self.predict_x, axis=1)

    def predict_x(self, x):
        """Return predictions for a vector"""
        
        a = self.alpha_y.values
        b = np.array([kernel_fun(support_vec, x) for _, support_vec in self.support_vectors_.iterrows()])

        return np.vdot(a,b)
    

    def compute_kernel(self, X):
        n = X.shape[0]
        self.K = np.zeros((n, n))
        for i in range(n):
            row1 = X[i]
            for j in range(n):
                row2 = X[j]
                self.K[i, j] = kernel_fun(row1, row2)


    def transform_dual(self, y):
        # Here, we keep lambda as the variable. We need to make A bigger to account for the 2 inequalities on lambda
        n = len(self.K_train_index)
        P = self.K[self.K_train_index].T[self.K_train_index]
        q = - y.values * np.ones(len(y))
        G = np.zeros((2 * n, n))
        h = np.zeros(2 * n)
        G[:n, :] = matrix(np.diag(y))
        G[n:2 * n, :] = matrix(-np.diag(y))
        h[:n] = self.C * np.ones(n)
        return matrix(P), matrix(q), matrix(G), matrix(h)