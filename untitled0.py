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
from sklearn.model_selection import train_test_split
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
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
        if j>700:
            break
    except KeyError:
        continue


for i in range(len(list_)):
    list_[i] = list_[i].reshape(1,list_[i].shape[0],list_[i].shape[1],list_[i].shape[2])
y_values_ = y_values[0:701]    
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

import tensorflow as tf

def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  input_layer = tf.reshape(features["x"], [-1, 480, 640, 3])

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=3,
      kernel_size=[8, 8],
      padding="same",
      strides=(8, 8),
      activation=tf.nn.relu)
  print(conv1)
  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
  print(pool1)
  # Convolutional Layer #2 and Pooling Layer #2
  ##########################################ERROR#########################
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=3,
      kernel_size=[2, 2],
      padding="same",
      strides=(1, 1),
      activation=tf.nn.relu)
  print(conv2)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=1)
  print(pool2)


  # Dense Layer
  pool2_flat = tf.reshape(pool2, [-1, 29*39*3 ])
  dense = tf.layers.dense(inputs=pool2_flat, units=120, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=4)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": logits,
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.mean_squared_error(labels, logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
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
  
  
# Create the Estimator
mnist_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="/home/vicky/mnist_convnet_model")

# Set up logging for predictions
tensors_to_log = {"probabilities": "softmax_tensor"}

logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=50)

# Train the model
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_train},
    y=Y_train,
    batch_size=1,
    num_epochs=None,
    shuffle=True)

# train one step and display the probabilties
mnist_classifier.train(
    input_fn=train_input_fn,
    steps=1,
    hooks=[logging_hook])

mnist_classifier.train(input_fn=train_input_fn, steps=1000)

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_test},
    y=Y_test,
    num_epochs=1,
    shuffle=False)

eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
print(eval_results)