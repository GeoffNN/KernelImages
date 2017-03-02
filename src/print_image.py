# coding: utf-8

# In[1]:

# remove warnings
import warnings

warnings.filterwarnings('ignore')
# ---
# get_ipython().magic('matplotlib inline')
import pandas as pd

pd.options.display.max_columns = 100
from matplotlib import pyplot as plt
import matplotlib

matplotlib.style.use('ggplot')
import seaborn as sns

sns.set(color_codes=True)
import numpy as np
# from PIL import Image, ImageDraw

pd.options.display.max_rows = 100

# ## Import Data

# In[3]:

train = pd.read_csv('../data/Xtr.csv', sep=",", header=None)
trainOutput = pd.read_csv('../data/Ytr.csv', sep=",")
test = pd.read_csv('../data/Xte.csv', sep=",", header=None)

# In[122]:

train.drop(train.columns[len(train.columns) - 1], axis=1, inplace=True)
test.drop(test.columns[len(test.columns) - 1], axis=1, inplace=True)

# In[125]:

test

# In[126]:

combined = train.append(test)
combined.reset_index(inplace=True)
combined.drop('index', inplace=True, axis=1)

# In[ ]:




# In[164]:
#
# index = 16
#
# list_of_pixels = np.array(
#     [(combined.iloc[index][i] + combined.iloc[index][i + 1024] + combined.iloc[index][i + 2048]) for i in range(1024)])
# list_of_pixels -= min(list_of_pixels)
# list_of_pixels *= 255 / max(list_of_pixels)
# # Do something to the pixels...
# im2 = Image.new("F", (32, 32))
# im2.putdata(list_of_pixels)
#
# # In[165]:
#
# im2.resize((200, 200)).show()


# In[ ]:
