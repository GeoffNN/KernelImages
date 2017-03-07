import numpy as np
from scipy.sparse import csr_matrix

# l = np.array([1, 2, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0])
# m = np.array([5, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
#
# ls = csr_matrix(l)
# ms = csr_matrix(m)


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
