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
import time

input = np.load("resnet_features.npy")
print(input.shape)
X = np.reshape(input,(input.shape[0],input.shape[1],input.shape[2]*input.shape[3]))
print(X.shape)

y_violent = np.zeros(87)
y_non_violent = np.ones(88)
y = np.append(y_violent,y_non_violent)
print(y.shape)


X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42, test_size=0.2)
print(X_train.shape,X_test.shape,y_train.shape, y_test.shape)


model = Sequential()
model.add(CuDNNLSTM(50, input_shape=(X.shape[1],X.shape[2]), return_sequences=False, kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(1,activation='sigmoid'))
model.summary()


optimizer = optimizers.adam(lr=0.001,decay=0.004)
model.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=["accuracy"])
start = time.time()
model.fit(X_train,y_train, epochs=20, verbose=1, validation_data=(X_test,y_test), batch_size=32)
end = time.time()
time = end - start
print("Time: ", time)

# model.save_weights("resnet_LSTM.h5")

# pred = model.predict(X_test)
# prediction = []
# for p in pred:
#     if p>=.5:
#         prediction.append(1)
#     else:
#         prediction.append(0)
# print(classification_report(prediction, y_test))
