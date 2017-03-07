
# coding: utf-8

# In[2]:

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
# from PIL import Image, ImageDraw
import pickle


pd.options.display.max_rows = 100


# ## Import Data

# In[3]:

train = pd.read_csv('../data/Xtr.csv', sep = ",", header = None)
trainOutput = pd.read_csv('../data/Ytr.csv', sep = ",")
test = pd.read_csv('../data/Xte.csv', sep = ",", header = None)


# In[4]:

train.drop(train.columns[len(train.columns)-1], axis=1, inplace=True)
test.drop(test.columns[len(test.columns)-1], axis=1, inplace=True)


# In[5]:

combined = train.append(test)
combined.reset_index(inplace=True)
combined.drop('index',inplace=True,axis=1)


# In[6]:

# images = []
# for index in range(len(combined)):
#     a = np.zeros((32,32, 3))
#     for j in range(32):
#         for k in range(32):
#             a[j,k] = (combined.iloc[index][j+32*k],combined.iloc[index][j+32*k+1024],combined.iloc[index][j+32*k+2048])
            
#     if index%(len(combined)/10) == 0:
#         print(index)
            
#     images.append(a)
#     pickle.dump(images, open('../pickles/images.pickle','wb'))


# In[7]:

images = pickle.load(open("../pickles/images.pickle", 'rb'))


# ## Histogram of Gradients pour UN channel d'UNE image (faudra boucler)

# In[110]:

def convolve2D(image, cvmatrix): #cvmatrix square, odd length and height matrix
    
    csize = int(cvmatrix.shape[0]/2)
    convolvedImage = np.array(image, dtype=complex)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            cvmatsubset = cvmatrix[max(0,csize-i):cvmatrix.shape[0]-max(0,(i+csize+1)-image.shape[0]),max(0,csize-j):cvmatrix.shape[1]-max(0,(j+csize+1)-image.shape[1])]
            imagesubset = image[max(0,i-csize):min(image.shape[0],i+csize+1),max(0,j-csize):min(image.shape[1],j+csize+1)]
#             print(i,j)
#             print(cvmatsubset)
#             print(imagesubset)
            convolvedImage[i,j] = np.sum(np.multiply(cvmatsubset,imagesubset))
#             print(convolvedImage[i,j])
#             print()
                
    return convolvedImage


# In[225]:

def HOG(image, blockSize = 4, nbins = 9):
    scharr = np.array([[ -3-3j, 0-10j,  +3 -3j],
                        [-10+0j, 0+ 0j, +10 +0j],
                        [ -3+3j, 0+10j,  +3 +3j]]) # Gx + j*Gy

    grad = convolve2D(image, scharr)
    hist = np.zeros((8, 8, 9))
    blockSize = 4
    nbins = 9
    
    for i in range(8):
        for j in range(8):
            for case in np.nditer(grad[i*blockSize:(i+1)*blockSize, j*blockSize:(j+1)*blockSize]):
                hist[i,j,int(abs(nbins*((360+np.angle(case, deg = True))%360)/360))] += np.absolute(case)
                
    return hist, grad



# ## Creation of total pickle

# In[295]:

def createHOG(images, blockSize = 4, nbins = 9):
    
    HOGfile = np.empty([nbins*images[0].shape[0]*2*3])
    for i in range(len(images)):
        line = np.empty([0])
        for r in range(3):
            hist, grad = HOG(images[i][:,:,r], blockSize = blockSize, nbins = nbins)
            line = np.append(line, np.ravel(hist))
        HOGfile = np.vstack((HOGfile, line))
        
    HOGfile = pd.DataFrame(HOGfile[1:]) #First line added I don't know why, but the other ones work

    pickle.dump(images, open('../pickles/HOG.pickle','wb'))
    HOGfile[:len(train)].to_csv('../pickles/HOG_train.csv', sep = ",", index=False, header = False)
    HOGfile[len(train):].to_csv('../pickles/HOG_test.csv', sep = ",", index=False, header = False)
    
    return HOGfile


# In[296]:

hf = createHOG(images)


# In[ ]:



