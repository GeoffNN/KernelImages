from tools import load_sparse_csr
from kernel_functions import *

histo_16s_train = load_sparse_csr("../data/histo_16s_train.npz")

# print(type(histo_16s_train))
# print(histo_16s_train)
K = kernel_matrix(histo_16s_train, generalized_histogram_kernel, a=0.5, b=0.5)
