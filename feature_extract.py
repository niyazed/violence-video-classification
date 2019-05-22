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
import time


model = VGG19(input_shape=(224,224,3),weights='imagenet', include_top=False)
new_model = model.output
new_model = Model(input=model.input,output=new_model)
new_model.summary()

v_path = "violent/"
nv_path = "non_violent/"
violent_frames = []
non_violent_frames = []
for frame in os.listdir(v_path):
    frame = cv2.imread(os.path.join(v_path,frame))
    violent_frames.append(frame)


for frame in os.listdir(nv_path):
    frame = cv2.imread(os.path.join(nv_path,frame))
    non_violent_frames.append(frame)



violent_frames = np.asarray(violent_frames)
violent_frames.shape
violent_vid = []
non_violent_vid = []
i = 0
while i < len(violent_frames):
    violent_vid.append(violent_frames[i:i+30])
    i = i+30
violent_vid = np.asarray(violent_vid)





i = 0
while i < len(non_violent_frames):
    non_violent_vid.append(non_violent_frames[i:i+30])
    i = i+30
non_violent_vid = np.asarray(non_violent_vid)



print(violent_vid.shape,non_violent_vid.shape)



y_violent = np.zeros(len(violent_vid))
y_non_violent = np.ones(len(non_violent_vid))



X = np.vstack((violent_vid,non_violent_vid))
y = np.append(y_violent,y_non_violent)


# In[ ]:


features = []

start = time.time()

for i in X:
    f = new_model.predict(i,batch_size=10,verbose=1)
    features.append(f)

end = time.time()
time = end - start
print("Time: ", time)
# In[ ]:


features = np.asarray(features)


# In[25]:


features = np.reshape(features,(features.shape[0],features.shape[1],features.shape[2]*features.shape[3], features.shape[4]))


# In[26]:


# np.save("resnet_features.npy",features)


# In[27]:


features.shape

