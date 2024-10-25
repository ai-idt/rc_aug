"""
This is our implementation of random convolutions as an image augmentation strategy.

For our paper, we left both n_layers and the kernel size k at their default values (3)
"""
import numpy as np
import tensorflow as tf
import random

def randConv_linear(input_shape,n_layers=3,k=3):
    img_in = tf.keras.layers.Input(shape=input_shape)
    img_ = tf.keras.layers.Conv3D(filters=1,kernel_size=k,strides=1,padding="same",kernel_initializer=tf.keras.initializers.HeNormal())(img_in) 
    for _ in range(1,n_layers):
        img_ = tf.keras.layers.Conv3D(filters=1,kernel_size=k,strides=1,padding="same",kernel_initializer=tf.keras.initializers.HeNormal())(img_)

    return tf.keras.Model(img_in,img_)

#This is an example code to perform random convolution on an input image
#Note that in our paper, we normalized images into [0;1] range before RC
#In this example, we assume the variable img to be the input numpy array of shape (in_shape)


if random.random() < augmentation_probability: #augmentation_probability is a hyperparameter we set to 0.5 in our paper; i.e., only half of images are augmented
    rc_mdl = randConv_linear(input_shape=img.shape)
    rc_image = rc_mdl(np.expand_dims(img,axis=[0,-1])) #Tf/keras expects images to have a batch and channel dimension, hence the expand_dims()
    img = np.squeeze(rc_image.numpy())
    img -= img.min() #Renorm RC image to [0;1]
    img /= img.max()