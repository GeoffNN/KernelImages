# coding: utf-8

from scipy.spatial.distance import pdist, squareform
import scipy

# ## Compute Gaussian Kernel

# sigma = 60

def gaussianKernel(combined, sigma):
    pairwise_sq_dists = squareform(pdist(combined, 'sqeuclidean'))
    return scipy.exp(-pairwise_sq_dists / sigma**2)

