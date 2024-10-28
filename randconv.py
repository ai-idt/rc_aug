"""
This is our implementation of random convolutions as an image augmentation strategy.

randConv_linear adapted from: https://openaccess.thecvf.com/content/CVPR2023/html/Choi_Progressive_Random_Convolutions_for_Single_Domain_Generalization_CVPR_2023_paper.html
randConv_nonlinear adapted from: https://ieeexplore.ieee.org/document/9961940/
"""
import numpy as np
import tensorflow as tf
import random

def randConv_linear(input_shape,n_layers=3,k=3,n_filters=1):
    #For our paper, we used all paramaters (n_layers, k, n_filters) at their default values
    img_in = tf.keras.layers.Input(shape=input_shape)
    img_ = tf.keras.layers.Conv3D(filters=n_filters,kernel_size=k,strides=1,padding="same",kernel_initializer=tf.keras.initializers.HeNormal())(img_in) 
    for _ in range(1,n_layers-1):
        img_ = tf.keras.layers.Conv3D(filters=n_filters,kernel_size=k,strides=1,padding="same",kernel_initializer=tf.keras.initializers.HeNormal())(img_)
    img_ = tf.keras.layers.Conv3D(filters=1,kernel_size=k,strides=1,padding="same",kernel_initializer=tf.keras.initializers.HeNormal())(img_)

    return tf.keras.Model(img_in,img_)

def randConv_nonlinear(input_shape,n_layers=4,k=3,n_filters=2):
    #For our paper, we used all paramaters (n_layers, k, n_filters) at their default values
    img_in = tf.keras.layers.Input(shape=input_shape)
    img_ = tf.keras.layers.Conv3D(filters=n_filters,kernel_size=k,strides=1,padding="same",kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0,stddev=1.))(img_in) 
    img_ = tf.keras.layers.LeakyReLU()(img_)
    for _ in range(1,n_layers-1):
        img_ = tf.keras.layers.Conv3D(filters=n_filters,kernel_size=k,strides=1,padding="same",kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0,stddev=1.))(img_) 
        img_ = tf.keras.layers.LeakyReLU()(img_)
    img_ = tf.keras.layers.Conv3D(filters=1,kernel_size=k,strides=1,padding="same",kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0,stddev=1.))(img_) 

    return tf.keras.Model(img_in,img_)

#This is an example code to perform linear random convolution on an input image
#Note that in our paper, we normalized images into [0;1] range before RC
#In this example, we assume the variable img to be the input numpy array of shape (in_shape), i.e., it has a channel dimension in the end (axis=-1)
if random.random() < augmentation_probability: #augmentation_probability is a hyperparameter we set to 0.5 in our paper; i.e., only half of images are augmented
    rc_mdl = randConv_linear(input_shape=img.shape)
    rc_image = rc_mdl(np.expand_dims(img,axis=0)) #Tf/keras expects images to have a batch and channel dimension, hence the expand_dims()
    img = np.squeeze(rc_image.numpy())
    img[img != 0] -= img[img != 0].min() #Renorm RC image to [0;1]
    img /= img.max()

#This is an example for the randConv_nonlinear (GIN) augmentation
gin_mdl = randConv_nonlinear(input_shape=img.shape)
gin_image = gin_mdl(np.expand_dims(img,axis=0)) #Tf/keras expects images to have a batch and channel dimension, hence the expand_dims()
gin_img = np.squeeze(gin_image.numpy())
gin_img[gin_img != 0] -= gin_img[gin_img != 0].min() #Renorm GIN image to [0;1]
gin_img /= gin_img.max()
#GIN performs a "mixup" interpolation between the original image and its GIN-augmented version
alpha_ = random.uniform(0.,1.)
img = (alpha_ * gin_img) + ((1. - alpha_) * img)