import numpy as np


def gaussian(x, y, gamma):
    return np.exp(-gamma * (x - y).norm())


def laplacian(x, y, a, b, rho):
    s = np.sum(abs(x ** a - y ** a) ** b)
    return np.exp(-rho * s)
