# coding: utf-8

import numpy as np
import pandas as pd
from src.kernel_svm import *


class multiclassSVM:
    
    def __init__(self, K=None, kernel_fun=None, C=300, **kwargs):
        self.kernel_fun = kernel_fun
        self.parameters = kwargs
        self.C = C
        self.K = K
        self.models = {}
        self.classes = None

    def fit(self, X_train, y_train):
        
        classes = np.unique(y_train)
        self.classes = classes
        for l1 in range(len(classes)):
            for l2 in range(l1+1,len(classes)):
                X_train_local = X_train[(y_train == classes[l1]) + (y_train == classes[l2])]
                y_train_local = y_train[(y_train == classes[l1]) + (y_train == classes[l2])]
                y_train_local = 2*(y_train_local == classes[l1]) - 1
                self.models[classes[l1],classes[l2]] = KernelSVM(K=self.K, kernel_fun=self.kernel_fun, C=self.C)
                self.models[classes[l1],classes[l2]].fit(X_train_local, y_train_local, K_train_index = y_train_local.index)
        
        
    def predict(self, X_test, X_test_index = None):
        
        ensemble_predictions = np.empty([len(X_test), 0])
        for k in self.models.keys():
            local_prediction = self.models[k].predict(X_test, X_test_index)
            ensemble_predictions = np.hstack((ensemble_predictions, np.array([k[0]*(local_prediction == 1) + k[1]*(local_prediction == -1)]).T))
            
        sum_array = np.array([np.sum(ensemble_predictions == c, axis = 1) for c in self.classes]).T
        final_predictions = self.classes[np.argmax(sum_array, axis=1)]
            
        return final_predictions, ensemble_predictions, sum_array



