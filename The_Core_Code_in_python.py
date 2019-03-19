
# coding: utf-8

# ## Importing libraries

# In[2]:


import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import os
import cv2
import math
import scipy
from scipy import ndimage
from tensorflow.python.framework import ops
from sklearn.model_selection import train_test_split
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


# ## Defining loss function

# In[3]:


def loss_1(labels, logits,batch_size):
    labels = tf.to_float(labels)
    sum=tf.zeros([1,], dtype=tf.float32)
    for i in range(batch_size):
        cx_label = tf.divide(tf.add(labels[i][0],labels[i][1]),2)
        cy_label = tf.divide(tf.add(labels[i][2],labels[i][3]),2)
        cx_logits = tf.divide(tf.add(logits[i][0],logits[i][1]),2)
        cy_logits = tf.divide(tf.add(logits[i][2],logits[i][3]),2)
        c_dist=tf.add(tf.square(tf.subtract(cx_label, cx_logits)),tf.square(tf.subtract(cy_label, cy_logits)))
        bx_label=tf.subtract(labels[i][1],labels[i][0])
        bx_logits=tf.subtract(logits[i][1],logits[i][0])
        hy_label=tf.subtract(labels[i][3],labels[i][2])
        hy_logits=tf.subtract(logits[i][3],logits[i][2])
        b_diff=tf.square(tf.subtract(bx_label, bx_logits))
        h_diff=tf.square(tf.subtract(hy_label, hy_logits))
        loss=tf.add(tf.add(b_diff, h_diff), c_dist)
        sum=sum+loss
    return sum/batch_size


# ## The CNN model

# In[4]:


def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  input_layer = tf.reshape(features["x"], [-1, 480, 640, 3])

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=8,
      use_bias=True,
      bias_initializer=tf.zeros_initializer(),
      kernel_size=[1, 1],
      padding="same",
      strides=(1, 1),
      activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=16,
      kernel_size=[1, 1],
      padding="same",
      strides=(1, 1),
      activation=tf.nn.relu)


  conv3 = tf.layers.conv2d(
      inputs=conv2,
      filters=16,
      kernel_size=[2, 2],
      padding="same",
      use_bias=True,
      bias_initializer=tf.zeros_initializer(),
      strides=(2, 2),
      activation=tf.nn.relu)


  conv4 = tf.layers.conv2d(
      inputs=conv3,
      filters=24,
      kernel_size=[1, 1],
      padding="same",
      strides=(1, 1),
      use_bias=True,
      bias_initializer=tf.zeros_initializer(),
      activation=tf.nn.relu)


  conv5 = tf.layers.conv2d(
      inputs=conv4,
      filters=24,
      kernel_size=[2, 2],
      padding="same",
      strides=(2, 2),
      use_bias=True,
      bias_initializer=tf.zeros_initializer(),
      activation=tf.nn.relu)


#       conv6 = tf.layers.conv2d(
#           inputs=conv5,
#           filters=48,
#           kernel_size=[1, 1],
#           padding="same",
#           strides=(1, 1),
#           use_bias=True,
#           bias_initializer=tf.zeros_initializer(),
#           activation=tf.nn.relu)

  pool3 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=2)

  conv7 = tf.layers.conv2d(
      inputs=pool3,
      filters=48,
      kernel_size=[2, 2],
      padding="same",
      strides=(2, 2),
      activation=tf.nn.relu,
      use_bias=True,
      bias_initializer=tf.zeros_initializer())

  conv8 = tf.layers.conv2d(
      inputs=conv7,
      filters=24,
      kernel_size=[1, 1],
      padding="same",
      strides=(1, 1),
      activation=tf.nn.relu,
      use_bias=True,
      bias_initializer=tf.zeros_initializer())

  conv9 = tf.layers.conv2d(
      inputs=conv8,
      filters=48,
      kernel_size=[1, 1],
      padding="same",
      strides=(1, 1),
      activation=tf.nn.relu,
      use_bias=True,
      bias_initializer=tf.zeros_initializer())

  conv10 = tf.layers.conv2d(
      inputs=conv9,
      filters=48,
      kernel_size=[2, 2],
      padding="same",
      strides=(2, 2),
      activation=tf.nn.relu,
      use_bias=True,
      bias_initializer=tf.zeros_initializer())

  conv11 = tf.layers.conv2d(
      inputs=conv10,
      filters=48,
      kernel_size=[1, 1],
      padding="same",
      strides=(1, 1),
      activation=tf.nn.relu,
      use_bias=True,
      bias_initializer=tf.zeros_initializer())


  conv12 = tf.layers.conv2d(
      inputs=conv11,
      filters=48,
      kernel_size=[1, 1],
      padding="same",
      strides=(1, 1),
      activation=tf.nn.relu,
      use_bias=True,
      bias_initializer=tf.zeros_initializer())
    
  # Dense Layer
  conv12_flat = tf.reshape(conv12, [-1, 3840 ])
    
  dense = tf.layers.dense(inputs=conv12_flat, units=120, activation=tf.nn.relu)

  dropout = tf.layers.dropout(
      inputs=dense, rate=0.1, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=4)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "co-ordinates": tf.multiply(x = tf.convert_to_tensor(np.array([255.00],dtype = 'float32')), y = logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  #loss = tf.losses.mean_squared_error(labels, logits)
  loss = loss_1(labels,logits,batch_size=10)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=0.00001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "iou": tf.metrics.mean_iou(
          labels,logits,4)
  }

  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


# ##  1) Training the CNN model on the training dataset

# In[5]:



#loading the path of the image location
path = 'D:\Flipkart\im\images'

#listing the name of the images
listing = (os.listdir(path))

#reading the 'train.csv'
df = pd.read_csv('D:\Flipkart\\training.csv')

#setting the index with column 'image_name' for easier access
df = df.set_index(keys='image_name',drop= True)

#some list and dicts for further storage and processing
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
for k in range(0,13800,200):
    print(k)
#reading the images in the list_ for training

    j=0

    list_ = []
    for i in true_listing[k:]:
        try:
            list_.append(np.array(Image.open(path + '/' + i)))
            j+=1
            print(j, end=', ')
            if j>199:
                break
        except KeyError:
            continue
    
    #converting the list of arrays in to a single array
    #X_train values
    for i in range(len(list_)):
        list_[i] = list_[i].reshape(1,list_[i].shape[0],list_[i].shape[1],list_[i].shape[2])
    
    #Y_train values
    y_values_ = y_values[k:k+200]    
    for i in range(len(list_)):
        y_values_[i] = y_values_[i].reshape(1,y_values_[i].shape[0],) 
  
    ds_X = np.concatenate((list_[:]),axis = 0)
    ds_Y = np.concatenate((y_values_[:]),axis=0)
    X_train, X_test, Y_train, Y_test = train_test_split(ds_X, ds_Y, test_size=0.1, random_state=42)
    
    
    #idx = 15
    #plt.imshow(X_train[idx])
    #plt.plot(Y_train[idx][0],Y_train[idx][2],Y_train[idx][0],Y_train[idx][3],marker = 'o')
    #plt.plot(Y_train[idx][1],Y_train[idx][3],Y_train[idx][1],Y_train[idx][2],marker = 'o')
    
    
    X_train = X_train/255
    X_test = X_test/255
    Y_train = Y_train/255
    Y_test = Y_test/255
    Y_train = np.squeeze(Y_train)
    Y_test = np.squeeze(Y_test)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    Y_train = Y_train.astype('float32')
    Y_test = Y_test.astype('int32')
          
    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir='D:\\reg_training_1')
    
    # Set up logging for predictions
    tensors_to_log = {"co-ordinates": "softmax_tensor"}
    
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)
    
    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": X_train},
        y=Y_train,
        batch_size=10,     
        num_epochs=10,
        shuffle=True)
    
    # train one step and display the probabilties
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=1,
        hooks=[logging_hook])
    
    mnist_classifier.train(input_fn=train_input_fn, steps=200)
    
    #for evaluation of the trained model
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": X_test},
        y=Y_test,
        num_epochs=1,
        shuffle=False)
    #X = np.expand_dims(X_test[0],axis=0)
    
    
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)
    


# ## 2) Training the CNN model on the vertically flipped training dataset

# In[4]:


df_vflip = pd.read_csv('D:\Flipkart\\vfliptraining.csv')

for k in range(0,13800,200):
    print(k)
    #for flipped images uncomment for activation
    list_ = []
    y_values_ = []
    for i in range(k,k+200,1):
        first=np.array(df_vflip)[i]
        list_.append(cv2.flip(cv2.imread(path+'\\'+first[0]),0))
        y_values_.append(first[1:])
        if i == 199:
            break
    #print(len(list_))

    for i in range(len(list_)):
       
        list_[i] = list_[i].reshape(1,list_[i].shape[0],list_[i].shape[1],list_[i].shape[2])
    
    #Y_train values
 
    for i in range(len(y_values_)):
        y_values_[i] = y_values_[i].reshape(1,y_values_[i].shape[0],) 
        
    ds_X = np.concatenate((list_[:]),axis = 0)
    ds_Y = np.concatenate((y_values_[:]),axis=0)
    X_train, X_test, Y_train, Y_test = train_test_split(ds_X, ds_Y, test_size=0.1, random_state=42)
    
    
    #idx = 15
    #plt.imshow(X_train[idx])
    #plt.plot(Y_train[idx][0],Y_train[idx][2],Y_train[idx][0],Y_train[idx][3],marker = 'o')
    #plt.plot(Y_train[idx][1],Y_train[idx][3],Y_train[idx][1],Y_train[idx][2],marker = 'o')
    
    
    X_train = X_train/255
    X_test = X_test/255
    Y_train = Y_train/255
    Y_test = Y_test/255
    Y_train = np.squeeze(Y_train)
    Y_test = np.squeeze(Y_test)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    Y_train = Y_train.astype('float32')
    Y_test = Y_test.astype('int32')
          
    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="D:\\reg_training_1")
    
    # Set up logging for predictions
    tensors_to_log = {"co-ordinates": "softmax_tensor"}
    
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)
    
    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": X_train},
        y=Y_train,
        batch_size=10,     
        num_epochs=5,
        shuffle=True)
    
    # train one step and display the probabilties
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=1,
        hooks=[logging_hook])
    
    mnist_classifier.train(input_fn=train_input_fn, steps=200)
    
    #for evaluation of the trained model
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": X_test},
        y=Y_test,
        num_epochs=1,
        shuffle=False)
    #X = np.expand_dims(X_test[0],axis=0)
    
    
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


# ## 3) Training the CNN model on the horizontally flipped training dataset

# In[5]:


df_hflip = pd.read_csv('D:\Flipkart\\hfliptraining.csv')

for k in range(0,13800,200):
    print(k)
    #for flipped images uncomment for activation
    list_ = []
    y_values_ = []
    for i in range(k,k+200,1):
        first=np.array(df_hflip)[i]
        list_.append(cv2.flip(cv2.imread(path+'\\'+first[0]),1))
        y_values_.append(first[1:])
        if i == 199:
            break

    for i in range(len(list_)):
       
        list_[i] = list_[i].reshape(1,list_[i].shape[0],list_[i].shape[1],list_[i].shape[2])
    
    #Y_train values
 
    for i in range(len(y_values_)):
        y_values_[i] = y_values_[i].reshape(1,y_values_[i].shape[0],) 
        
    ds_X = np.concatenate((list_[:]),axis = 0)
    ds_Y = np.concatenate((y_values_[:]),axis=0)
    X_train, X_test, Y_train, Y_test = train_test_split(ds_X, ds_Y, test_size=0.1, random_state=42)
    
    X_train = X_train/255
    X_test = X_test/255
    Y_train = Y_train/255
    Y_test = Y_test/255
    Y_train = np.squeeze(Y_train)
    Y_test = np.squeeze(Y_test)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    Y_train = Y_train.astype('float32')
    Y_test = Y_test.astype('int32')
          
    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="D:\\reg_training_1")
    
    # Set up logging for predictions
    tensors_to_log = {"co-ordinates": "softmax_tensor"}
    
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)
    
    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": X_train},
        y=Y_train,
        batch_size=10,     
        num_epochs=5,
        shuffle=True)
    
    # train one step and display the probabilties
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=1,
        hooks=[logging_hook])
    
    mnist_classifier.train(input_fn=train_input_fn, steps=200)
    
    #for evaluation of the trained model
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": X_test},
        y=Y_test,
        num_epochs=1,
        shuffle=False)
    #X = np.expand_dims(X_test[0],axis=0)
    
    
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)
    


# ## Getting predition out of the trained model on the test images

# In[80]:


#Loading the test set and getting the results

df1 = pd.read_csv('D:\Flipkart\\test.csv') #Dataframe containing the test-data
test_list=np.array(df1)[:, 0] #Extracting the names of the images


for i in range(0, len(test_list), 50): #Predicting 
    print(i)
    test_list1=test_list[i:i+50]
    batch=len(test_list1)
    input_X=np.zeros((batch, 480, 640, 3))
    
    for j in range(len(test_list1)):
        input_X[j, :, :, :]=cv2.imread(path + '\\' + test_list1[j])
    
    input_X=input_X.astype('float32')/255
    
    pred_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": input_X},
    shuffle=False)

    #calling the prediction function
    predict = mnist_classifier.predict(input_fn = pred_input_fn)
    
    #unwrapping the predict iterator
    cls_pred = np.array(list(predict))
    pred=np.zeros((batch, 4))
    
    for j in range(len(test_list1)):
        pred[j, :]=cls_pred[j]['co-ordinates']
        if pred[j, 0]<0:
            pred[j, 0]=1
        if pred[j, 1]>640:
            pred[j, 1]=639
        if pred[j, 2]<0:
            pred[j, 2]=1
        if pred[j ,3]>480:
            pred[j, 3]=479
            
    for j in range(len(test_list1)):
        df1.loc[i+j, 'x1']=pred[j][0]
        df1.loc[i+j, 'x2']=pred[j][1]
        df1.loc[i+j, 'y1']=pred[j][2]
        df1.loc[i+j, 'y2']=pred[j][3]
    
df1.to_csv('D:\Flipkart\\test_updated.csv',index=False)

