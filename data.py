import tensorflow as tf
import numpy as np
import pickle
import random
import copy
from imageio import imread
from glob import glob
import sys
import os
from model_config import BUFFER_SIZE,BATCH_SIZE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler 


def load_mnist(limit = None,anomaly=None,percentage_anomaly=0):
    """
        Loads the MNIST datasets

        limit (int) sets a limit on the number of test and training samples
        anomaly (int) is the anomalous class
        percentage_anomaly (float) adds a percentage of the anomalous/novel class to the training set
    """
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    if anomaly is not None:
        indicies = np.argwhere(train_labels == int(anomaly))
        sample_indicies = random.sample(list(indicies[:,0]), int(percentage_anomaly*
                                                                  indicies.shape[0]))

        mask_train  = np.invert(train_labels == int(anomaly))
        mask_train[sample_indicies] = True

        train_images = train_images[mask_train]
        train_labels = train_labels[mask_train]

    if limit is not None:
        train_images = train_images[:limit,...]
        train_labels = train_labels[:limit,...]  
        test_images  = test_images[:limit,...]
        test_labels  = test_labels[:limit,...] 

    train_images = process(train_images.reshape(train_images.shape[0], 28, 28, 1))
    test_images = process(test_images.reshape(test_images.shape[0], 28, 28, 1))

    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    return (train_dataset,train_images, train_labels, test_images, test_labels)

def load_fashion_mnist(limit = None,anomaly=None,percentage_anomaly=0):
    """
        Loads the FMNIST dataset

        limit (int) sets a limit on the number of test and training samples
        anomaly (int) is the anomalous class
        percentage_anomaly (float) adds a percentage of the anomalous/novel class to the training set
    """
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

    if anomaly is not None:
        indicies = np.argwhere(train_labels == int(anomaly))
        sample_indicies = random.sample(list(indicies[:,0]), int(percentage_anomaly*
                                                                  indicies.shape[0]))

        mask_train  = np.invert(train_labels == int(anomaly))
        mask_train[sample_indicies] = True

        train_images = train_images[mask_train]
        train_labels = train_labels[mask_train]

    if limit is not None:
        train_images = train_images[:limit,...]
        train_labels = train_labels[:limit,...]  
        test_images  = test_images[:limit,...]
        test_labels  = test_labels[:limit,...] 

    train_images = process(train_images.reshape(train_images.shape[0], 28, 28, 1))
    test_images = process(test_images.reshape(test_images.shape[0], 28, 28, 1))

    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    return (train_dataset,train_images, train_labels, test_images, test_labels)

def load_cifar10(limit = None,anomaly=None,percentage_anomaly=0):
    """
        Loads the CIFAR10 dataset

        limit (int) sets a limit on the number of test and training samples
        anomaly (int) is the anomalous class
        percentage_anomaly (float) adds a percentage of the anomalous/novel class to the training set
    """
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    train_labels,test_labels = train_labels[:,0],test_labels[:,0] #because cifar labels are weird

    if anomaly is not None:
        indicies = np.argwhere(train_labels == int(anomaly))
        sample_indicies = random.sample(list(indicies[:,0]), int(percentage_anomaly*
                                                                  indicies.shape[0]))

        mask_train  = np.invert(train_labels == int(anomaly))
        mask_train[sample_indicies] = True

        train_images = train_images[mask_train]
        train_labels = train_labels[mask_train]

    if limit is not None:
        train_images = train_images[:limit,...]
        train_labels = train_labels[:limit,...]  
        test_images  = test_images[:limit,...]
        test_labels  = test_labels[:limit,...] 

    train_images = process(train_images.reshape(train_images.shape[0], 32, 32, 3))
    test_images = process(test_images.reshape(test_images.shape[0], 32, 32, 3))

    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    return (train_dataset,train_images, train_labels, test_images, test_labels)

def load_mvtec(limit = None,anomaly=None,percentage_anomaly=0):
    """
        Loads the MVTEC-AD dataset

        limit (int) sets a limit on the number of test and training samples
        anomaly (int) is the anomalous class
        percentage_anomaly (float) adds a percentage of the anomalous/novel class to the training set
    """
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    train_labels,test_labels = train_labels[:,0],test_labels[:,0] #because cifar labels are weird

    if anomaly is not None:
        indicies = np.argwhere(train_labels == int(anomaly))
        sample_indicies = random.sample(list(indicies[:,0]), int(percentage_anomaly*
                                                                  indicies.shape[0]))

        mask_train  = np.invert(train_labels == int(anomaly))
        mask_train[sample_indicies] = True

        train_images = train_images[mask_train]
        train_labels = train_labels[mask_train]

    if limit is not None:
        train_images = train_images[:limit,...]
        train_labels = train_labels[:limit,...]  
        test_images  = test_images[:limit,...]
        test_labels  = test_labels[:limit,...] 

    train_images = process(train_images.reshape(train_images.shape[0], 32, 32, 3))
    test_images = process(test_images.reshape(test_images.shape[0], 32, 32, 3))

    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    return (train_dataset,train_images, train_labels, test_images, test_labels)

def get_mvtec_images(directory='dataset/MVTecAD/'):
    """"
        Walks through MVTEC dataset and returns data in the same structure as tf
    """

    output = [] 
    data_names = [dI for dI in os.listdir(directory) if os.path.isdir(os.path.join(directory,dI))]

    # Remove grid, screw and zipper as they are greyscale and incompatible with our models 
    for remove_names in ['grid','screw','zipper']:
        data_names.remove(remove_names)    

   

def rgb2gray(rgb):
    return np.expand_dims(np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140]),axis=-1)

def process(data):
    """
        Scales data between 0 and 1 on a per image basis

        data (np.array) is either the test or training data

    """
    output = copy.deepcopy(data).astype('float32')
    for i,image in enumerate(data):
        x,y,z = image.shape
        output[i,...] = MinMaxScaler(feature_range=(0,1)
                                      ).fit_transform(image.reshape([x*y,z])).reshape([x,y,z])
    return output


