import numpy as np
from sparse_function import sparse_absolute, sparse_min, sparse_norm_2


def linear_kernel(x, y):
    res = x.dot(y.transpose()).data[0]
    return res


def polynomial_kernel(x, y, d):
    res = (x.dot(y.transpose()).data[0] + 1) ** d
    return res


def non_gaussian_rbf_kernel(x, y, a, b, gamma):
    res = sparse_absolute(x.power(a) - y.power(a))
    res = res.power(b)
    res = res.sum()
    res = np.exp(-gamma * res)
    return res


def generalized_histogram_kernel(x, y, a, b):
    x_a = x.power(a)
    y_b = y.power(b)
    res = sparse_min(x_a, y_b)
    res = res.sum()
    return res


def gaussian_kernel(x, y, gamma):
    res = np.exp(-gamma * sparse_norm_2(x - y, True))
    return res


def kernel_matrix(X, kernel, **kwargs):
    n = X.shape[0]
    K = np.zeros((n, n))
    for i in range(n):
        print(i)
        for j in range(n):
            if j % 1000 == 0:
                print('----------------', j)
            K[i, j] = kernel(X[i], X[j], **kwargs)
    return K


def scipy_sparse_to_spmatrix(A):
    coo = A.tocoo()
    SP = spmatrix(coo.data.tolist(), coo.row.tolist(), coo.col.tolist(), size=A.shape)
    return SP
