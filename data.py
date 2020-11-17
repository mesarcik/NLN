import tensorflow as tf
import numpy as np
import pickle
import random
import sys
from model_config import BUFFER_SIZE,BATCH_SIZE
from sklearn.model_selection import train_test_split


def load_mnist(limit = None,anomaly=None,percentage_anomaly=0):
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

    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype(np.float32)

    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype(np.float32)

    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    return (train_dataset,train_images, train_labels, test_images, test_labels)

def load_fashion_mnist(limit = None,anomaly=None,percentage_anomaly=0):
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

    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype(np.float32)

    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype(np.float32)

    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    return (train_dataset,train_images, train_labels, test_images, test_labels)

def load_cifar10(limit = None,anomaly=None,percentage_anomaly=0):
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

    train_images = train_images.reshape(train_images.shape[0], 32, 32, 3).astype(np.float32)

    test_images = test_images.reshape(test_images.shape[0], 32, 32, 3).astype(np.float32)

    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    return (train_dataset,train_images, train_labels, test_images, test_labels)

