from print_image import print_img
import pandas as pd
from math import floor

data_train = pd.read_csv('../data/Xtr.csv', sep=',',header = None)
data_train = data_train.iloc[:,:3072]

print_img(data_train.iloc[0,:])