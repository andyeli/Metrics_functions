# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 16:16:20 2021

@author: Andrea
"""

import numpy as np
from sklearn.svm import SVC
from scipy.io import loadmat
import imbalance_metrics as im
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#%%
# indian_pines = loadmat('Indian_pines_corrected.mat')
# dataset = indian_pines['indian_pines_corrected']
# gt2 = loadmat('Indian_pines_gt.mat')
# ground_truth = gt2['indian_pines_gt']


# dataset2 = np.load('Dataset/S2_TrainingData.npy')
# dataset = dataset2[53,:,:,:]
# gt2 = np.load('Dataset/S2_TrainingLabels.npy')
# ground_truth = gt2[53,:,:]
#dataset = dataset/10000

salinas = loadmat('SalinasA_corrected.mat')
dataset = salinas['salinasA_corrected']
gt2 = loadmat('SalinasA_gt.mat')
ground_truth = gt2['salinasA_gt']

height = dataset.shape[0]
width = dataset.shape[1]
bands = dataset.shape[2]

dataset = dataset/10000


#%% S V M
x = np.reshape(dataset, (height*width, bands))
y = np.reshape(ground_truth, -1)

result = np.where(y != 0)[0]
x = x[result,:]
y =y[result]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.90)

svclassifier = SVC(kernel='linear')
svclassifier.fit(x_train, y_train)

y_pred = svclassifier.predict(x_test)

Cm = confusion_matrix(y_test, y_pred)

print(Cm)
print(classification_report(y_test,y_pred))
#%%
multiclass_metrics = im.report(y_test,y_pred)
