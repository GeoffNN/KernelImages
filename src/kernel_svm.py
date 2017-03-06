import numpy as np
import cvxopt
from cvxopt import matrix, solvers
import pandas as pd

rho = 1.
a = 0.25
b = 1


def kernel(x, y):
    s = np.sum(abs(x ** a - y ** a) ** b)
    return np.exp(-rho * s)


def kernel_matrix(X):
    n = X.shape[0]
    K = np.zeros((n, n))
    for i in range(n):
        if (i % 500 == 0):
            print(i)
        for j in range(n):
            K[i, j] = kernel(X.iloc[i, :], X.iloc[j, :])
    return K


def transform_dual(X, y, K, C=300, ):
    # Here, we keep lambda as the variable. We need to make A bigger to account for the 2 inequalities on lambda
    n = X.shape[0]
    X_m = matrix(X.values)
    Q = matrix(np.diag(y)) * K * matrix(np.diag(y))
    p = - np.ones(n)
    G = np.zeros((2  n, n))
    h = np.zeros(2 * n)
    G[:n, :n] = np.eye(n)
    G[n:2 * n, :n] = -np.eye(n)
    h[:n] = C * np.ones(n)
    return matrix(Q), matrix(p), matrix(G), matrix(h)


# def transform_dual2(X,y,C=.5):
#     # Here, we keep lambda as the variable. We need to make A bigger to account for the 2 inequalities on lambda
#     n = X.shape[0]
#     X_m = matrix(X.values)
#     P = X_m * X_m.T
#     q = - y * np.ones(n)
#     G = np.zeros((2 * n, n))
#     h = np.zeros(2 * n)
#     G[:n, :] = matrix(np.diag(y))
#     G[n:2 * n, :] = matrix(-np.diag(y))
#     h[:n] = C * np.ones(n)
#     return matrix(P), matrix(q), matrix(G), matrix(h)

# Computation of SVM
# Output : alpha.T * Diag(y)

def svm(X: object, y: object, C: object = 300) -> object:
    Q, p, G, h = transform_dual(X, y, C)
    w_sol = solvers.qp(Q, p, G, h)
    print(w_sol)
    return np.dot(w_sol['x'].T, np.diag(y))


def predict(w, x, K):
    ker = np.array([kernel(x, X[i]) for i in range(X.shape[0])])
    return np.sign(w * ker)


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
