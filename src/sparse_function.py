import numpy as np


def sparse_absolute(x):
    res = x.copy()
    temp = np.abs(res.data)
    res.data = temp
    return res


def sparse_scalar_multiplication(x, a):
    res = x.copy()
    res.data = a * res.data
    return res


def sparse_min(x, y):
    res = sparse_absolute(x + y) - sparse_absolute(x - y)
    res = sparse_scalar_multiplication(res, 0.5)
    return res


def sparse_norm_2(x, square_norm=False):
    res = x.power(2).sum()
    if square_norm:
        return res
    else:
        return np.sqrt(res)
