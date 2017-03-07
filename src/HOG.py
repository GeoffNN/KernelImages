
# coding: utf-8

# In[1]:

# remove warnings
import warnings
warnings.filterwarnings('ignore')
# ---
get_ipython().magic('matplotlib inline')
import pandas as pd
pd.options.display.max_columns = 100
from matplotlib import pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import seaborn as sns
sns.set(color_codes=True)
import numpy as np
from PIL import Image, ImageDraw
import pickle


pd.options.display.max_rows = 100


# ## Import Data

# In[2]:

train = pd.read_csv('../data/Xtr.csv', sep = ",", header = None)
trainOutput = pd.read_csv('../data/Ytr.csv', sep = ",")
test = pd.read_csv('../data/Xte.csv', sep = ",", header = None)


# In[3]:

train.drop(train.columns[len(train.columns)-1], axis=1, inplace=True)
test.drop(test.columns[len(test.columns)-1], axis=1, inplace=True)


# In[4]:

combined = train.append(test)
combined.reset_index(inplace=True)
combined.drop('index',inplace=True,axis=1)


# In[5]:

# images = []
# for index in range(len(combined)):
#     a = np.zeros((32,32, 3))
#     for j in range(32):
#         for k in range(32):
#             a[j,k] = (combined.iloc[index][j+32*k],combined.iloc[index][j+32*k+1024],combined.iloc[index][j+32*k+2048])
            
#     if index%(len(combined)/10) == 0:
#         print(index)
            
#     images.append(a)
#     pickle.dump(images, open('images.pickle','wb'))


# In[6]:

images = pickle.load(open("images.pickle", 'rb'))


# ## Histogram of Gradients pour UN channel d'UNE image (faudra boucler)

# In[7]:

from scipy import signal

def HOG(image):
    scharr = np.array([[ -3-3j, 0-10j,  +3 -3j],
                        [-10+0j, 0+ 0j, +10 +0j],
                        [ -3+3j, 0+10j,  +3 +3j]]) # Gx + j*Gy

    grad = signal.convolve2d(image, scharr, boundary='symm', mode='same')
    hist = np.zeros((8, 8, 9))

    for i in range(8):
        for j in range(8):
            for case in np.nditer(grad[i*4:(i+1)*4, j*4:(j+1)*4]):
                hist[i,j,int(abs(9*((360+np.angle(case, deg = True))%360)/360))] += np.absolute(case)
                
    return hist


# In[8]:

images[0][:,:,0].shape


# In[9]:

hist = HOG(images[0][:,:,0])


# In[ ]:



