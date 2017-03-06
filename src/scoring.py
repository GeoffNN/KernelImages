from scipy.stats import binom
import pandas as pd


def get_train_val(training, train_frac=.75, seed=0):
    sorter = pd.Series(binom(train_frac, seed=seed).rvs(len(training)), index=training.index)

    train = training[sorter == 0]
    val = training[sorter == 1]
    return train, val


def get_score(y_pred, y_groundtruth):
    return (y_pred == y_groundtruth).mean()
