import tensorflow as tf
import numpy as np
import pickle
import random
import copy
from imageio import imread
from glob import glob
from skimage.transform import resize 
from tqdm import tqdm
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

def load_mvtec(SIMO_class,limit = None,percentage_anomaly=0):
    """
        Loads the MVTEC-AD dataset

        SIMO_class (str) is the SIMO class
        limit (int) sets a limit on the number of test and training samples
        percentage_anomaly (float) adds a percentage of the anomalous/novel class to the training set
    """
    (train_images, train_labels), (test_images, test_labels) = get_mvtec_images(SIMO_class)

    if limit is not None:
        train_images = train_images[:limit,...]
        train_labels = train_labels[:limit,...]  
        test_images  = test_images[:limit,...]
        test_labels  = test_labels[:limit,...] 

    train_images = process(train_images)
    test_images = process(test_images)

    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    return (train_dataset,train_images, train_labels, test_images, test_labels)

def get_mvtec_images(SIMO_class, directory='datasets/MVTecAD/',  dim = (256,256,3) ):
    """"
        Walks through MVTEC dataset and returns data in the same structure as tf
        This is typical of MISO detection. 
    """
    if (('grid' in SIMO_class) or
        ('screw' in SIMO_class) or 
        ('zipper' in SIMO_class)): 
        dim = (dim[0],dim[1],1)

    train_images, test_images, train_labels ,test_labels = [], [], [], []
    
    # if the training dataset has already been created then return that
    print('{}.pickle loaded'.format(SIMO_class))
    if os.path.exists('{}/{}.pickle'.format(directory,SIMO_class)):
        with open('{}/{}.pickle'.format(directory,SIMO_class),'rb') as f:
            return pickle.load(f)

    print('Creating data for {}'.format(SIMO_class))
    for f in tqdm(glob("{}/{}/train/good/*.png".format(directory,SIMO_class))): 
        img = imread(f)
        train_images.append(resize(img, dim ,anti_aliasing=False))
        train_labels.append('non_anomalous')

    for f in tqdm(glob("{}/{}/test/*/*.png".format(directory,SIMO_class))): 
        img = imread(f)
        test_images.append(resize(img, dim ,anti_aliasing=False))
        if 'good' in f: 
            test_labels.append('non_anomalous')
        else:
            test_labels.append('bottle')

    pickle.dump(((np.array(train_images), np.array(train_labels)),(np.array(test_images), np.array(test_labels))),
                open('{}/{}.pickle'.format(directory,SIMO_class), 'wb'), protocol=1)

    return (np.array(train_images), np.array(train_labels)), (np.array(test_images), np.array(test_labels))

            

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


