import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix


def load_train_feats(path_to_data="../data/"):
    X = pd.read_csv(path_to_data + 'Xtr.csv', sep=",", header=None)
    return X.drop(X.columns[-1], axis=1)


def load_test_feats(path_to_data="../data/"):
    X = pd.read_csv(path_to_data + "Xte.csv", sep=",", header=None)
    return X.drop(X.columns[-1], axis=1)


def load_train_target(path_to_data="../data/"):
    return pd.read_csv(path_to_data + "data/Ytr.csv", sep=",", index_col=0)['Prediction']


def print_results(Y_pred, title, path_to_results="results/"):
    Y_pred.to_csv(path_to_results + title)


def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])
