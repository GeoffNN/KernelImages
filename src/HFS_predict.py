
# coding: utf-8

# In[1]:

# remove warnings
import warnings
warnings.filterwarnings('ignore')
# ---

import sys
sys.path.append('./lib')


get_ipython().magic('matplotlib inline')
import pandas as pd
pd.options.display.max_columns = 100
from matplotlib import pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import seaborn as sns
sns.set(color_codes=True)
import numpy as np

from lib.graph_init import *
from lib.hard_hfs import *

pd.options.display.max_rows = 100


# ## Import Data

# In[2]:

train = pd.read_csv('../data/Xtr.csv', sep = ",", header = None)
trainOutput = pd.read_csv('../data/Ytr.csv', sep = ",")
test = pd.read_csv('../data/Xte.csv', sep = ",", header = None)


# In[3]:

train.drop(train.columns[len(train.columns)-1], axis=1, inplace=True)
test.drop(test.columns[len(test.columns)-1], axis=1, inplace=True)


# In[5]:

combined = train.append(test)
combined.reset_index(inplace=True)
combined.drop('index',inplace=True,axis=1)


# In[ ]:




# ## HFS 
# 0 is unlabeled, but was a class: that's why we use the +1/-1 trick

# In[26]:

input_vect = combined
target_vect = np.array(trainOutput['Prediction'])
target_vect = np.append(target_vect+1,[0 for i in range(len(combined) - len(target_vect))])


# In[27]:

lp = LaplacianParams()

sim = build_graph(input_vect, GraphParams())



# In[28]:

L = build_laplacian(sim,lp)


# In[29]:

hfs0, confidence = simple_hfs(input_vect, target_vect, L, sim)


# In[63]:

IdTest = np.array([i for i in range(1, 1 + len(combined) - len(train))])
output = [int(x) for x in hfs0[IdTest + len(train) - 1] - 1]
df_output = pd.DataFrame()
df_output['Id'] = IdTest
df_output['Prediction'] = output
df_output[['Id','Prediction']].to_csv('../predictions/test_hfs.csv', sep = ",", index=False)



