#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 19:24:33 2018

@author: anamika
"""
#%%
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from numpy import*
import cv2
from PIL import Image
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras import backend as K
K.set_image_dim_ordering('th')
path1="D:\My_PROJECT\Input"
path2="D:\My_PROJECT\In"
rows,cols=50,50

#%%
imlist=os.listdir(path2)
im1=np.array(Image.open(path2+"/"+imlist[0])).flatten()  
immatrix = np.array([np.array(Image.open(path2+"/"+ im2)).flatten()
              for im2 in imlist]) 

#%%
label=np.ones((len(imlist)),dtype=int)
label[0:97]=0
label[97:226]=1
label[226:]=2
     
data,label=shuffle(immatrix,label,random_state=2)
train_data=[data,label]

#%%batch size to train
batch=10
classes=3
epochs=20

#number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 3
# convolution kernel size
nb_conv = 3
#%%
(x, y) = (train_data[0],train_data[1])
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
x_train = x_train.reshape(x_train.shape[0], 1, 50,50)
x_test = x_test.reshape(x_test.shape[0], 1, 50, 50)
x_train=x_train.astype('float32')
x_test=x_test.astype('float32')
x_train/=255
x_test/=255

#%% convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, classes)
Y_test = np_utils.to_categorical(y_test, classes)


#%%
model=Sequential()

# conv=>relu==>POOL
model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='valid',
                        input_shape=(1, rows, cols)))
convout1 = Activation('relu')
model.add(convout1)
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
convout2 = Activation('relu')
model.add(convout2)
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))

#conv==>relu==>POOL
model.add(Convolution2D(64, nb_conv, nb_conv,
                        border_mode='valid',
                        input_shape=(1, rows, cols)))
convout1 = Activation('relu')
model.add(convout1)
model.add(Convolution2D(64, nb_conv, nb_conv))
convout2 = Activation('relu')
model.add(convout2)
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

#first set of fc==> RELU layer
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))

#softmax classifier
model.add(Dense(classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adadelta')
#%%
#mod = model.fit(x_train, Y_train, batch_size=batch, nb_epoch=epochs,
             # verbose=1, validation_data=(x_test, Y_test))
            
mod = model.fit(x_train, Y_train, batch_size=batch, nb_epoch=epochs,
               verbose=1, validation_split=0.2)


#%%     
model.save("D:\My_PROJECT\MYModel")


#%%
from keras.models import load_model
import cv2
import numpy as np
from PIL import Image

CLASSES=['ball','car','cow']
model=load_model("D:\My_PROJECT\MYModel")
test_path="D:\My_PROJECT\TestingData"
files=os.listdir(test_path)  
for file in files:
    image=cv2.imread(test_path+"/"+file)
    #cv2.imshow("ff",image)
    img=cv2.resize(image,(50,50))
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img=np.array(gray).flatten()
    #print(img.shape)
    img=np.array(img)
    img=img.reshape(1,1,50,50) 
    img=img.astype('float32')
    img/=255   
#classes = model.predict_classes(img)
    prob = model.predict(img)[0]
    
    for i in range(prob.shape[0]):
        if prob[i]>0.1:
            label = "{}: {:.2f}%".format(CLASSES[i],
	               	prob[i] * 100)
            print(file,": ",label)
            cv2.putText(image, label, (30*i+8, 60*i+35), cv2.FONT_HERSHEY_SIMPLEX,
		         0.8, (255, i*255, 255), 1,cv2.LINE_AA) ## 1st arg: image on which to write
                                         ## 2nd : what to write
                                         ## 3rd :starting point of text
                                         ## 4th :font style
                                         ## 5th : font Size
                                         ## 6th : color of text
                                         ## 7th : text thickness
                                         ## 8th : type of line used
             #cv2.rectangle(image,(50,50),(200,200),(255,0,0),2)
    cv2.imshow(file, image)
    if cv2.waitKey(0) & 0xFF == 27:
        cv2.destroyAllWindows()
#cv2.imshow("gg",image)
    

    
