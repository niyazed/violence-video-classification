from keras.applications import *
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import numpy as np
from keras.models import Sequential
from keras.layers import *
import pylab
from keras.models import Model	
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import os
import cv2
import keras
import sklearn
import pandas
from time import time
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.utils import to_categorical
from keras.models import load_model
from keras.layers import *
from keras import layers
from keras import Model
from keras.callbacks import TensorBoard
from keras import optimizers
import matplotlib.pyplot as plt
from keras.applications import *
from sklearn.metrics import classification_report



labels=['violent','non_violent']


# Train dataset loading and Stacking both label

X_train = np.load('E:/Niloy/Journal/frames/train/'+labels[0] + '.npy')
y_train = np.zeros(X_train.shape[0])

print(X_train.shape)

for i, label in enumerate(labels[1:]):
    x = np.load('E:/Niloy/Journal/frames/train/'+label + '.npy')
    X_train = np.vstack((X_train, x))
    y_train = np.append(y_train, np.full(x.shape[0], fill_value= (i + 1)))

print(X_train.shape)
# Test dataset loading and  Stacking both label

X_test = np.load('E:/Niloy/Journal/frames/test/'+labels[0] + '.npy')
y_test = np.zeros(X_test.shape[0])


for i, label in enumerate(labels[1:]):
    x = np.load('E:/Niloy/Journal/frames/test/'+label + '.npy')
    X_test = np.vstack((X_test, x))
    y_test = np.append(y_test, np.full(x.shape[0], fill_value= (i + 1)))
    
# print(test.shape)
# X_train = X_train.astype('float32')/255
# X_test = X_test.astype('float32')/255

model = resnet50.ResNet50(input_shape=(224,224,3),weights='imagenet', include_top=False)
new_model = model.output
new_model = Model(input=model.input,output=new_model)
new_model.summary()


# features = []
X = np.vstack((X_train,X_test))
y = np.append(y_train,y_test)
# for i in X:
f = new_model.predict(X,batch_size=10,verbose=1)
    # features.append(f)


features = np.asarray(f)
print(features.shape)


# In[25]:


# features = np.reshape(features,(features.shape[0],features.shape[1],features.shape[2]*features.shape[3], features.shape[4]))


# In[26]:


np.save("resnet_features.npy",features)
np.save("y.npy",y)


# In[27]:


# features.shape

