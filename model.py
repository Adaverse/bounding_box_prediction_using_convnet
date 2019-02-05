import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import os
import math
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops

path = '/media/vicky/Akash Back Up/images'
listing = (os.listdir(path))
df = pd.read_csv('/home/vicky/Downloads/training.csv')

df = df.set_index(keys='image_name',drop= True)
true_listing = []
y_values = []
dict_={}

for i in listing:
    try:
        dict_[i] = pd.Series.tolist(df.loc[i])
        y_values.append(np.array(pd.Series.tolist(df.loc[i])))
        true_listing.append(i)
    except KeyError:
        continue
j=0
list_ = []
for i in true_listing:
    try:
        list_.append(np.array(Image.open(path + '/' + i)))
        j+=1
        print(j)
        if j>1000:
            break
    except KeyError:
        continue


for i in range(len(list_)):
    list_[i] = list_[i].reshape(1,list_[i].shape[0],list_[i].shape[1],list_[i].shape[2])
y_values_ = y_values[0:1001]    
for i in range(len(list_)):
    y_values_[i] = y_values_[i].reshape(1,y_values_[i].shape[0],1) 

X_train = np.concatenate((list_[:]),axis = 0)
Y_train = np.concatenate((y_values_[:]),axis=0)
idx = 8
plt.imshow(X_train[idx])
plt.plot(Y_train[idx][0],Y_train[idx][2],Y_train[idx][0],Y_train[idx][3],marker = 'o')
plt.plot(Y_train[idx][1],Y_train[idx][3],Y_train[idx][1],Y_train[idx][2],marker = 'o')


def create_placeholders(n_H0, n_W0, n_C0, n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_H0 -- scalar, height of an input image
    n_W0 -- scalar, width of an input image
    n_C0 -- scalar, number of channels of the input
    n_y -- scalar, number of classes
        
    Returns:
    X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"
    Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float"
    """

    X = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0])
    Y = tf.placeholder(tf.float32, [None, n_y])
    
    return X, Y

