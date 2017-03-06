import pandas as pd


def load_train_feats(path_to_data="data/"):
    X = pd.read_csv(path_to_data + 'Xtr.csv', sep=",", header=None)
    return X.drop(X.columns[-1], axis=1)


def load_test_feats(path_to_data="data/"):
    X = pd.read_csv(path_to_data + "data/Xte.csv", sep=",", header=None)
    return X.drop(X.columns[-1], axis=1)


def load_train_target(path_to_data="data/"):
    return pd.read_csv(path_to_data + "data/Ytr.csv", sep=",", index_col=0)['Prediction']


def print_results(Y_pred, title, path_to_results="predictions/"):
    Y_pred.to_csv(path_to_results + title)
