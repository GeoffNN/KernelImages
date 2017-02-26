import numpy as np
from numpy.linalg import inv

from graph_init import GraphParams, LaplacianParams, build_graph, build_laplacian


# TODO: refactor using fit/transform

def hfs(X, Y, graph_params=GraphParams(), laplacian_params=LaplacianParams(), mode='simple', **kwargs):
    if mode is 'simple':
        return simple_hfs(X, Y, graph_params, laplacian_params)
    elif mode is 'iterative':
        if 't' in kwargs.keys():
            T = kwargs['t']
            return iterative_hfs(X, Y, graph_params, laplacian_params, T=T)
        return iterative_hfs(X, Y, graph_params, laplacian_params)
    elif mode is 'online':
        return online_hfs(X, Y, graph_params, laplacian_params)


def simple_hfs(X, Y, L, W):
    n_samples = len(X)
    #n_classes = len(np.unique(Y)) - 1
    n_classes = max(Y)

    # compute linear target for labelled samples
    l_idx = np.nonzero(Y)[0]
    u_idx = np.nonzero(Y == 0)[0]
    n_l = len(l_idx)
    y = -np.ones((n_l, n_classes))
    for i in range(n_l):
        y[i, int(Y[l_idx[i]] - 1)] = 1
    # Compute solution
    f_l = y
    #W = build_graph(X, graph_params)
    #L = build_laplacian(W, laplacian_params)
    f_u = inv(L[[[x] for x in u_idx], u_idx]).dot(W[[[x] for x in u_idx], l_idx]).dot(f_l)

    # Compute label assignment
    l_l = f_l.argmax(axis=1)
    l_u = f_u.argmax(axis=1)
    labels = np.zeros(n_samples)
    labels[l_idx] = l_l
    labels[u_idx] = l_u
    
    confidence = np.vstack([np.zeros(n_samples) for i in range(int(n_classes))])
    confidence[:,l_idx] = f_l.T
    confidence[:,u_idx] = f_u.T
    
    return labels + 1, confidence.T


def soft_hfs(X, Y, c_l, c_u, L):
    n_samples = len(X)
    n_classes = len(np.unique(Y))
    gamma = 0.001

    # compute linear target for labelled samples
    l_idx = np.nonzero(Y)[0]
    u_idx = np.nonzero(Y == 0)[0]
    n_l = len(l_idx)
    y = -np.ones((n_samples, n_classes))
    for i in range(n_l):
        y[l_idx[i], int(Y[l_idx[i]] - 1)] = 1
    # Compute solution


    #W = build_graph(X, graph_params)
    #L = build_laplacian(W, laplacian_params)

    C = np.diag((c_l-c_u)*(Y != 0) + c_u)

    Q = L + gamma*np.eye(len(L))
    f = inv(inv(C).dot(Q) + np.eye(len(L))).dot(y)

    labels = f.argmax(axis=1)
    
    return labels



def iterative_hfs(X, Y, graph_params, laplacian_params, T=50):
    W = build_graph(X, graph_params)
    L = build_laplacian(W, laplacian_params)

    n_samples = len(X)
    n_classes = len(np.unique(Y)) - 1

    # compute linear target for labelled samples
    l_idx = np.nonzero(Y)[0]
    u_idx = np.nonzero(Y == 0)[0]
    n_l = len(l_idx)
    n_u = len(u_idx)
    y = -np.ones((n_samples, n_classes))
    for i in range(n_l):
        y[int(l_idx[i]), int(Y[l_idx[i]] - 1)] = 1

    D = W.sum(axis=0)
    f = y
    for t in range(T):
        for k in u_idx:
            f[k, :] = W[:, k].dot(f) / D[k]
    labels = f.argmax(axis=1) + 1
    return labels


def online_hfs(X, Y, graph_params, laplacian_params):
    pass
