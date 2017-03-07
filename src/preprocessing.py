from scipy.sparse import csr_matrix
from histo import data_to_histogram
from tools import load_train_feats, load_test_feats,save_sparse_csr

data_train = load_train_feats()
data_test = load_test_feats()

histo_16_train = data_to_histogram(data_train, 16)
histo_16_test = data_to_histogram(data_test, 16)

# Transform to scipy sparse matrix
histo_16s_train = csr_matrix(histo_16_train.values)
histo_16s_test = csr_matrix(histo_16_test.values)

# Store into numpy format

save_sparse_csr("../data/histo_16s_train", histo_16s_train)
save_sparse_csr("../data/histo_16s_test", histo_16s_test)