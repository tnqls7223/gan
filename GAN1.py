from configparser import Interpolation
from cv2 import CV_8U, INTER_AREA
import tensorflow as tf
import pandas as ps
from tensorflow import keras
import keras.backend as K
from keras.layers.core import Dense, Dropout, Flatten,Activation, Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D, SeparableConv2D, UpSampling2D
from keras.layers import BatchNormalization,Input
from keras.models import Model
from keras.optimizers import RMSprop
from keras.preprocessing.image import array_to_img
import numpy as np
from PIL import Image
import scipy as sp
import cv2
import os

import warnings ; warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from tqdm import tqdm



inputpic = cv2.imread("Chim.png")
inputpic = cv2.resize(inputpic,(28,28))
inputpic = cv2.cvtColor(inputpic,cv2.COLOR_BGR2GRAY,CV_8U)
inputpicn=np.array(inputpic)
inputpicn=np.reshape(inputpicn,(1,784))
inputpicn = inputpicn/255
print(inputpicn.shape)

img_path = './Data/'
imgList = os.listdir(img_path)
imgList_np = []
for i in imgList:
    imgs = cv2.imread(img_path+i)
    imgs = cv2.cvtColor(imgs,cv2.COLOR_BGR2GRAY,CV_8U)
    imgs = cv2.resize(imgs,(28,28))   
    img_array=np.array(imgs)
    imgList_np.append(img_array)

Xtrain=np.array(imgList_np)
Xtrain=np.reshape(Xtrain,(len(Xtrain),28,28,1))
Xtrain=Xtrain / 255
print(Xtrain.shape)

random_noise=np.random.normal(0, 1, (1, 100))

print(random_noise.shape)

##plt.imshow(Xtrain[0], cmap='gray')
##plt.show()

##plt.imshow(inputpic, cmap='gray')
##plt.show()





################################################3
disc_input = Input(shape=(28, 28, 1))

x = Conv2D(filters=64, kernel_size=5, strides=2, padding='same')(disc_input)
x = Activation('relu')(x)
x = Dropout(rate=0.4)(x)

x = Conv2D(filters = 64, kernel_size=5, strides=2, padding='same')(x)
x = Activation('relu')(x)
x = Dropout(rate=0.4)(x)

x = Conv2D(filters=128, kernel_size=5, strides=2, padding='same')(x)
x = Activation('relu')(x)
x = Dropout(rate=0.4)(x)

x = Conv2D(filters=128, kernel_size=5, strides=1, padding='same')(x)
x = Activation('relu')(x)
x = Dropout(rate=0.4)(x)

x = Flatten()(x)
disc_output = Dense(units=1, activation='sigmoid', kernel_initializer='he_normal')(x)

discriminator = Model(disc_input, disc_output)
discriminator.summary()
###############################################################################################


gen_dense_size=(7, 7, 64)

gen_input = Input(shape = (784, ))
x = Dense(units=np.prod(gen_dense_size))(gen_input)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Reshape(gen_dense_size)(x)

x = UpSampling2D()(x)
x = Conv2D(filters=128, kernel_size=5, padding='same', strides=1)(x)
x = BatchNormalization(momentum=0.9)(x)
x = Activation('relu')(x)

x = UpSampling2D()(x)
x = Conv2D(filters = 64, kernel_size=5, padding='same', strides=1)(x)
x = BatchNormalization(momentum=0.9)(x)
x = Activation('relu')(x)

x = Conv2D(filters=64, kernel_size=5, padding='same', strides=1)(x)
x = BatchNormalization(momentum=0.9)(x)
x = Activation('relu')(x)

x = Conv2D(filters=1, kernel_size=5, padding='same', strides=1)(x)
gen_output = Activation('sigmoid')(x)

generator = Model(gen_input, gen_output)
generator.summary()


discriminator.compile(optimizer=RMSprop(lr=0.0008), loss='binary_crossentropy', metrics=['accuracy'])
discriminator.trainable = False
model_input = Input(shape=(784, ))
model_output = discriminator(generator(model_input))
model = Model(model_input, model_output) 

model.compile(optimizer=RMSprop(lr=0.0004), loss='binary_crossentropy', metrics=['accuracy'])


def train_discriminator(Xtrain, batch_size):
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    
    idx = np.random.randint(0, len(Xtrain), batch_size)
    true_imgs = Xtrain[idx]
    discriminator.fit(true_imgs, valid, verbose=0)
    
    noise = np.random.normal(0, 1, (batch_size, 784))
    gen_imgs = generator.predict(noise)
    
    discriminator.fit(gen_imgs, fake, verbose=0)


def train_generator(batch_size):
    valid = np.ones((batch_size, 1))
    noise = np.random.normal(0, 1, (batch_size, 784))
    model.fit(noise, valid, verbose=1)

for epoch in tqdm(range(3000)):
    train_discriminator(Xtrain, 64)
    train_generator(64)

random_noise=np.random.normal(0, 1, (1, 784))
##gen_result=generator.predict(random_noise)
gen_result=generator.predict(inputpicn)
gen_img=array_to_img(gen_result[0])
model.save('S-datam.h5')
generator.save('S-datag.h5')
discriminator.save('S-datad.h5')
plt.imshow(gen_img, cmap='gray')
plt.show()
