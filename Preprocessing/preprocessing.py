import pandas as pd
from scipy.sparse import csr_matrix
from histo import data_to_histogram

data_train = pd.read_csv('../data/Xtr.csv', sep=',', header=None)
data_test = pd.read_csv('../data/Xte.csv', sep=',', header=None)

# Delete the last column
data_train = data_train.iloc[:, :3072]
data_test = data_test.iloc[:, :3072]

histo_16_train = data_to_histogram(data_train, 16)
histo_16_test = data_to_histogram(data_test, 16)

# Transform to sparse dataframe
histo_16s_train = csr_matrix(histo_16_train.values)
histo_16s_test = csr_matrix(histo_16_test.values)

# Store into pickles
histo_16s_train.to_pickle("../data/histo_16s_train")
histo_16s_test.to_pickle("../data/histo_16s_test")
