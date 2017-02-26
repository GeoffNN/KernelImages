import numpy as np
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from scipy.linalg import sqrtm, inv

"""This code is based on Daniel Calandriello's code for Michal Valko's Graphs in ML course (2016) at MVA - ENS Cachan."""


class GraphTypeError(Exception):
    pass


class GraphParams:
    def __init__(self, graph_type='knn', thresh=None, sigma2=1):
        self.type = graph_type
        self.thresh = thresh
        if thresh is None:
            if graph_type is 'knn':
                self.thresh = 10
            elif graph_type is 'eps':
                self.thresh = 1
        if graph_type not in ['knn', 'eps']:
            raise GraphTypeError("Not a valid graph graph_type")
        self.sigma2 = sigma2


class LaplacianParams:
    def __init__(self, normalization='unn', gamma=.05):
        self.normalization = normalization
        self.gamma = gamma


def build_graph(X, graph_params=GraphParams(), metric='euclidean'):
    """Builds a graph (knn or epsilon) weight matrix W
    W is sparse - to be optimized somehow
    """
    graph_type = graph_params.type
    sigma2 = graph_params.sigma2
    graph_thresh = graph_params.thresh
    n = len(X)
    W = np.zeros((n, n))
    if graph_type is 'knn':
        D = kneighbors_graph(X, graph_thresh, metric=metric, mode='distance').toarray()
    elif graph_type is 'eps':
        graph_thresh = -sigma2 * np.log(graph_thresh)
        D = radius_neighbors_graph(X, graph_thresh, metric=metric, mode='distance').toarray()
    W[D > 0] = np.exp(-D[D > 0] / sigma2)
    return W


def build_laplacian(W, laplacian_params):
    normalization = laplacian_params.normalization
    gamma = laplacian_params.gamma
    D = np.diag(np.sum(W != 0,1))
    if normalization is 'unn':
        L = D - W
    elif normalization is 'rw':
        L = np.eye(len(W)) - inv(D).dot(W)
    elif normalization is 'sym':
        d = inv(sqrtm(D))
        L = np.eye(len(W)) - d.dot(W).dot(d)
    return L + gamma * np.eye(len(L))
