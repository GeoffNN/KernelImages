from numpy.random import permutation
import pandas as pd


def get_train_val(training, train_frac=.75, seed=0):
    # Classes are balanced in trainset so no need to rebalance
    n_train = int(len(training)*train_frac)
    sorter = pd.Series(permutation([0 for i in range(n_train)] + [1 for i in range(len(training)-n_train)]), index=training.index)
    train = training[sorter == 0]
    val = training[sorter == 1]
    return train, val, sorter


def get_score(y_pred, y_groundtruth):
    return (y_pred == y_groundtruth).mean()
