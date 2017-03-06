import pandas as pd
import numpy as np
from histo import data_to_histo

data_train = pd.read_csv('../data/Xtr.csv', sep=',',header = None)
data_test = pd.read_csv('../data/Xte.csv', sep=',',header = None)


# Delete the last column
data_train = data_train.iloc[:,:3072]
data_test = data_test.iloc[:,:3072]


histo_16_train = data_to_histo(data_train,16)
histo_16_test = data_to_histo(data_test,16)


histo_16_train_sparse = histo_16_train.to_sparse(fill_value = 0)
histo_16_test_sparse = histo_16_test.to_sparse(fill_value = 0)


histo_16_train_sparse.to_pickle("../data/histo_16s_train")
histo_16_test_sparse.to_pickle("../data/histo_16s_test")