
# coding: utf-8

# In[5]:



import tensorflow as tf
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
import numpy as np
import argparse
import random
import pickle
import cv2
import os
import glob
import os.path as path
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD
from keras import backend as K
K.set_image_data_format('channels_first')
from keras.models import load_model
import picamera
import h5py
path='/home/pi/'
camera=picamera.PiCamera()
camera.vflip=True
camera.hflip=True

camera.resolution=(400,550)
camera.start_preview()
a=0
for i in range(0,10):
    camera.capture(str(a)+".jpg")
    camera.capture('raspistill -w 64 -h 64 -q 75 -br 100 -ex -ISO 200 -roi 0.5.5,0.5,0.16,0.12 -o image.jpg')
    a=a+1
camera.stop_preview()
#print(mat)
test_mat=[]
model=load_model('model.h5')
for i in range(0,5):
    img = cv2.imread(str(i)+'.jpg')
    img = cv2.resize(img,(32,32))
    img_ = np.asarray(img)
    img_ = np.rollaxis(img_,-1)
    test_mat.append(img_)
test_mat=np.array(test_mat)
#print(test_mat.shape)
pred_= model.predict(test_mat,verbose=1)
print(pred_)
prob=[]
cls=[]
cls_num=[]
cls_max_count=[]
for i in range(0,5):
    max_val=max(pred_[i])
    prob.append('%.4f' % max(pred_[i]))
    for j in range(0,3):
        if pred_[i][j]==max_val:
            cls_num.append(j)
            cls.append('class '+str(j))
#print(pred_[3])
pred_[pred_ > .5] = 1
pred_[pred_<0.5]=0
#print(pred_)
print("image number")
print(prob)
print(cls_num)
for c in range(0,3):
    cls_max=cls_num.count(c)
    cls_max_count.append(cls_max)
print(cls_max_count)
max_cls_pred=cls_max_count.index(max(cls_max_count))
print('class'+str(max_cls_pred))

