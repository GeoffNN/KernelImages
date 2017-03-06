import numpy as np
import pandas as pd


# Compute the index in the histogram of pixel x
def get_ind(x, i, n_bins):
    r, g, b = x[i] + 0.5, x[i + 1024] + 0.5, x[i + 2048] + 0.5
    ind = n_bins * n_bins * int(b * n_bins) + n_bins * int(g * n_bins) + int(r * n_bins)
    return ind


# Compute the histogram of one RGB image
def rgb_to_histogram(img_src, n_bins):
    histogram = np.zeros((n_bins * n_bins * n_bins))

    for i in range(1024):
        ind = get_ind(img_src, i, n_bins)
        histogram[ind] += 1

    return pd.DataFrame(histogram.T)


# Transform a data frame of RGB image in a df of histograms
def data_to_histogram(d, n_bins):
    df = pd.DataFrame()
    for i in range(d.shape[0]):
        h = rgb_to_histogram(d.iloc[i, :], n_bins)
        df = df.append(h.T)
    return df
