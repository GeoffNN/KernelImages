# coding: utf-8

# In[1]:
import pandas as pd
import pickle


from src.multiclass_svm import multiclassSVM
from src.HOG import *
from src.generate_kernel import gaussianKernel


print("\nImporting data...", end=" ")


train = pd.read_csv('./data/Xtr.csv', sep = ",", header = None)
test = pd.read_csv('./data/Xte.csv', sep = ",", header = None)

trainOutput = pd.read_csv('./data/Ytr.csv', sep = ",")

train.drop(train.columns[len(train.columns)-1], axis=1, inplace=True)
test.drop(test.columns[len(test.columns)-1], axis=1, inplace=True)
combined = train.append(test)
combined.reset_index(inplace=True)
combined.drop('index',inplace=True,axis=1)

print("Done \nCreating Histogram Of Gradients...", end=" ")


combined_HOG = createHOG(storeImages(combined))

print("Done \nGenerating Gaussian kernel of the HOG...", end=" ")

sigma = 60
K = gaussianKernel(combined_HOG, sigma)

print("Done \nMaking predictions with multiclass SVM...", end=" ")

newtrain = combined_HOG[:5000]
newtest = combined_HOG[5000:]
newtrainOutput = trainOutput['Prediction']

mSVM = multiclassSVM(K = K)

mSVM.fit(newtrain, newtrainOutput)

final_predictions, _, _ = mSVM.predict(newtest, newtest.index)

print("Done \nDumping to CSV...", end=" ")

IdTest = np.array([i for i in range(1, 1 + len(test))])
output = [int(x) for x in final_predictions]
df_output = pd.DataFrame()
df_output['Id'] = IdTest
df_output['Prediction'] = output
df_output[['Id','Prediction']].to_csv('./Yte.csv', sep = ",", index=False)

print("Done \n\nComputation Done!\n\n")


