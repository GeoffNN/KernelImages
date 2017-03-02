import numpy as np
import cvxopt
from cvxopt import matrix, solvers
import pandas as pd


def transform_dual(X, y, C=.5):
    # Here, we keep lambda as the variable. We need to make A bigger to account for the 2 inequalities on lambda
    d = len(X)
    n = X.shape[0]
    Q = np.diag(y) * X * X.T * np.diag(y)
    p = np.ones(n)
    G = np.zeros((2 * n, n))
    h = np.zeros(2 * n)
    G[:n, :n] = np.eye(n)
    G[n:2 * n, :n] = -np.eye(n)
    h[:n] = C * np.ones(n)
    return Q, p, G, h


def SVM(X, y, C=.5):
    Q, p, G, h = transform_dual(X, y, C)
    w_sol = solvers.qp(Q, p, G, h)
    return w_sol


def predict(w, yval):
    return np.sign(w, yval)


def onevsallSVM(X, y, C=.5):
    models = {}
    for cls in np.unique(y):
        X_train = pd.concat((X.loc[y == cls], X.loc[y != cls]))
        y_train = pd.Series([0 if yval == cls else 1 for yval in y])
        models[cls] = SVM(X_train, y_train, C=.5)


def onevsoneSVM(X, y, C=.5):
    for class1 in np.unique(y):
        for class2 in np.unique(y):
            if class1 != class2:
                X_train = pd.concat((X.loc[y == class1], X.loc[y == class2]))
                y_train = pd.concat((y.loc[y == class1]), y.loc(y == class2))
