import numpy as np
from tools import load_sparse_csr
from kernel_functions import kernel_matrix, non_gaussian_rbf_kernel, generalized_histogram_kernel
import pandas as pd

histo_16s_train = load_sparse_csr("../data/histo_16s_train.npz")

K_gen_histo = kernel_matrix(histo_16s_train, generalized_histogram_kernel, a=0.5, b=0.5)
K_gen_df = pd.DataFrame(K_gen_histo)
np.save("generalized_histo_kernel", K_gen_histo)
K_gen_df.to_pickle("gen_histo_ker")

K_non_gaussian_rbf = kernel_matrix(histo_16s_train, non_gaussian_rbf_kernel, a=0.25, b=1, gamma=1)
K_non_gauss_df = pd.DataFrame(K_non_gaussian_rbf)
np.save("non_gaussian_kernel", K_non_gaussian_rbf)
K_non_gauss_df.to_pickle("non_gauss_ker")
