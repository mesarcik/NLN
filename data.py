import tensorflow as tf
import numpy as np
import pickle
import random
import sys
from model_config import BUFFER_SIZE,BATCH_SIZE,input_shape
from sklearn.model_selection import train_test_split
sys.path.insert(1,'/home/mmesarcik/phd/Workspace/lofar-dev/')
from DL4DI.preprocessor import preprocessor 


def load_mnist(limit = None,anomaly=None,percentage_anomaly=0):
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    if anomaly is not None:
        indicies = np.argwhere(train_labels == int(anomaly))
        #sample_indicies = random.sample(list(indicies[:,0]), int(percentage_anomaly*
        #                                                          indicies.shape[0]))

        mask_train  = np.invert(train_labels == int(anomaly))
        #mask_train[sample_indicies] = True

        train_images = train_images[mask_train]
        train_labels = train_labels[mask_train]

    if limit is not None:
        train_images = train_images[:limit,...]
        train_labels = train_labels[:limit,...]  
        test_images  = test_images[:limit,...]
        test_labels  = test_labels[:limit,...] 

    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
    train_images =  process(train_images,input_shape[0],input_shape[1],mag=False)

    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
    test_images= process(test_images,input_shape[0],input_shape[1],mag=False)

    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    return (train_dataset,train_images, train_labels, test_images, test_labels)

def load_fashion_mnist(limit = None,anomaly=None,percentage_anomaly=0):
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

    if anomaly is not None:
        indicies = np.argwhere(train_labels == int(anomaly))
        #sample_indicies = random.sample(list(indicies[:,0]), int(percentage_anomaly*
        #                                                          indicies.shape[0]))

        mask_train  = np.invert(train_labels == int(anomaly))
        #mask_train[sample_indicies] = True

        train_images = train_images[mask_train]
        train_labels = train_labels[mask_train]

    if limit is not None:
        train_images = train_images[:limit,...]
        train_labels = train_labels[:limit,...]  
        test_images  = test_images[:limit,...]
        test_labels  = test_labels[:limit,...] 

    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
    train_images =  process(train_images,input_shape[0],input_shape[1],mag=False)

    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
    test_images= process(test_images,input_shape[0],input_shape[1],mag=False)

    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    return (train_dataset,train_images, train_labels, test_images, test_labels)

def load_cifar10(limit = None,anomaly=None,percentage_anomaly=0):
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    train_labels,test_labels = train_labels[:,0],test_labels[:,0] #because cifar labels are weird

    if anomaly is not None:
        indicies = np.argwhere(train_labels == int(anomaly))
        #sample_indicies = random.sample(list(indicies[:,0]), int(percentage_anomaly*
        #                                                          indicies.shape[0]))

        mask_train  = np.invert(train_labels == int(anomaly))
        #mask_train[sample_indicies] = True

        train_images = train_images[mask_train]
        train_labels = train_labels[mask_train]

    if limit is not None:
        train_images = train_images[:limit,...]
        train_labels = train_labels[:limit,...]  
        test_images  = test_images[:limit,...]
        test_labels  = test_labels[:limit,...] 

    train_images = train_images.reshape(train_images.shape[0], 32, 32, 3)
    train_images =  process(train_images,input_shape[0],input_shape[1],mag=False)

    test_images = test_images.reshape(test_images.shape[0], 32, 32, 3)
    test_images= process(test_images,input_shape[0],input_shape[1],mag=False)

    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    return (train_dataset,train_images, train_labels, test_images, test_labels)

def load_hera(limit = None,anomaly=None,percentage_anomaly=0,_3db=True):
    path = '/home/mmesarcik/phd/Workspace/lofar-dev/DL4DI/data_generation/datasets/HERA_comp_1_2_01-09-2020.pkl'
    with open(path,'rb') as f: 
        x_train,_,labels,_,info = pickle.load(f)

    train_images, test_images, train_labels, test_labels = train_test_split(x_train, labels, test_size = 0.20)

    if anomaly is not None:
        indicies = np.argwhere(np.char.find(train_labels, anomaly)>=0)
        sample_indicies = random.sample(list(indicies[:,0]), int(percentage_anomaly*
                                                                  indicies.shape[0]))

        mask_train  = np.invert(np.char.find(train_labels, anomaly)>=0)
        mask_train[sample_indicies] = True

        train_images = train_images[mask_train]
        train_labels = train_labels[mask_train]

    if limit is not None:
        train_images= train_images[:10000,...]
        train_labels = train_labels[:10000,...]  
        test_images = test_images[:5000,...]
        test_labels  = test_labels[:5000,...] 

    train_images = process(train_images,input_shape[0],input_shape[1],mag=True)
    test_images = process(test_images, input_shape[0],input_shape[1],mag=True)
    #########################################################################################################
    # TODO remove this ugly code.
    test_labels = test_labels.astype('<U24')#to apppend to it
    if _3db:
        inds_test = random.sample(range(test_images.shape[0]),
                                  int(test_images.shape[0]*0.3))# 30% of the data
        for ind in inds_test:
            test_images[ind,:,16:24,:] = 0.001 * test_images[ind,:,16:24,:]
            test_labels[ind] = '{}_{}'.format(str(test_labels[ind]) , '3db')
    #########################################################################################################

    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    return (train_dataset,train_images, train_labels, test_images, test_labels)

def process(data,x,y,mag=False):
    p = preprocessor(data)
    p.interp(x,y)
    if mag: p.get_magnitude()
    p.minmax(per_baseline=True,feature_range=(0,1))
    return np.float32(p.get_processed_cube())

