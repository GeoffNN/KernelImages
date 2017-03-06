import numpy as np
<<<<<<< HEAD
<<<<<<< 1048b49e6e3481bf95d2987772e52b45a82a4757
=======


def kernel_lap(x,y,rho,a,b):
	sum = np.sum(abs(x**a-y**a)**b)
	return np.exp(-rho*sum)

>>>>>>> 2c267555d0d41cb89a6ea12257c17a558a0d2fb3
import cvxopt
from cvxopt import matrix, solvers
import pandas as pd


def transform_dual(X, y, C=.5):
    # Here, we keep lambda as the variable. We need to make A bigger to account for the 2 inequalities on lambda
    n = X.shape[0]
    X_m = matrix(X.values)
    Q = matrix(np.diag(y)) * X_m * X_m.T * matrix(np.diag(y))
    p = np.ones(n)
    G = np.zeros((2 * n, n))
    h = np.zeros(2 * n)
    G[:n, :n] = np.eye(n)
    G[n:2 * n, :n] = -np.eye(n)
    h[:n] = C * np.ones(n)
    return matrix(Q), matrix(p), matrix(G), matrix(h)


def SVM(X, y, C=.5):
    Q, p, G, h = transform_dual(X, y, C)
    w_sol = solvers.qp(Q, p, G, h)
    print(w_sol)
    return w_sol['x']


# TODO: correct predict
def predict(w, yval):
    return np.sign(w.dot(yval))


# TODO: finish the two pipelines
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
<<<<<<< HEAD
=======

def kernel_lap(x,y,rho,a,b):
	sum = np.sum(abs(x**a-y**a)**b)
	return np.exp(-rho*sum)

>>>>>>> preprocessing (creation des histos) + kernel svm (tourne mais pas efficace : grosse erreur +  temps de calcul du noyau trop long
=======

>>>>>>> 2c267555d0d41cb89a6ea12257c17a558a0d2fb3
