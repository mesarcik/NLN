import tensorflow as tf
import numpy as np
import random
from model_config import BUFFER_SIZE,BATCH_SIZE
from sklearn.model_selection import train_test_split
from utils.data import (get_mvtec_images, 
                        process,
                        rgb2gray,
                        get_patched_dataset,
                        random_rotation,
                        random_crop,
                        resize,
                        sizes)


def load_mnist(args):
    """
        Loads the MNIST datasets
        
        args (Namespace) Command line parameters from utils.cmd_input
    """
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_images = resize(np.expand_dims(train_images,axis=-1), (32, 32 ,1))
    test_images =  resize(np.expand_dims(test_images,axis=-1), (32, 32 ,1))

    if str(args.anomaly_class) is not None:
        if args.anomaly_type == 'MISO':
            indicies = np.argwhere(train_labels == int(str(args.anomaly_class)))
            sample_indicies = random.sample(list(indicies[:,0]), int(args.percentage_anomaly*
                                                                      indicies.shape[0]))

            mask_train  = np.invert(train_labels == int(str(args.anomaly_class)))
        else: 
            indicies = np.argwhere(train_labels != int(str(args.anomaly_class)))
            sample_indicies = random.sample(list(indicies[:,0]), int(args.percentage_anomaly*
                                                                      indicies.shape[0]))

            mask_train  = train_labels == int(str(args.anomaly_class))

        train_images = train_images[mask_train]
        train_labels = train_labels[mask_train]

    if args.limit is not None:
        train_images = train_images[:args.limit,...]
        train_labels = train_labels[:args.limit,...]  
        test_images  = test_images[:args.limit,...]
        test_labels  = test_labels[:args.limit,...] 

    if args.patches:
        (train_images, 
         train_labels, 
         test_images, 
         test_labels) = get_patched_dataset(train_images,
                                            train_labels,
                                            test_images,
                                            test_labels,
                                            (1,args.patch_x, args.patch_y, 1),
                                            (1,args.patch_stride_x, args.patch_stride_y, 1))


    # TODO
    # I need to note that this way of processing might be weird,
    train_images = process(train_images)
    test_images = process(test_images)


    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    return (train_dataset,train_images, train_labels, test_images, test_labels)

def load_fashion_mnist(args):
    """
        Loads the FMNIST dataset

        args (Namespace) Command line parameters from utils.cmd_input
    """
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    train_images = resize(np.expand_dims(train_images,axis=-1), (32, 32 ,1))
    test_images =  resize(np.expand_dims(test_images,axis=-1), (32, 32 ,1))

    if str(args.anomaly_class) is not None:
        if args.anomaly_type == 'MISO':
            indicies = np.argwhere(train_labels == int(str(args.anomaly_class)))
            sample_indicies = random.sample(list(indicies[:,0]), int(args.percentage_anomaly*
                                                                      indicies.shape[0]))

            mask_train  = np.invert(train_labels == int(str(args.anomaly_class)))
        else: 
            indicies = np.argwhere(train_labels != int(str(args.anomaly_class)))
            sample_indicies = random.sample(list(indicies[:,0]), int(args.percentage_anomaly*
                                                                      indicies.shape[0]))

            mask_train  = train_labels == int(str(args.anomaly_class))

        mask_train[sample_indicies] = True
        train_images = train_images[mask_train]
        train_labels = train_labels[mask_train]

    if args.limit is not None:
        train_images = train_images[:args.limit,...]
        train_labels = train_labels[:args.limit,...]  
        test_images  = test_images[:args.limit,...]
        test_labels  = test_labels[:args.limit,...] 

    if args.patches:
        data  = get_patched_dataset(train_images,
                                    train_labels,
                                    test_images,
                                    test_labels,
                                    p_size = (1,args.patch_x, args.patch_y, 1),
                                    s_size = (1,args.patch_stride_x, args.patch_stride_y, 1))

        train_images, train_labels, test_images, test_labels = data

    # TODO
    # I need to note that this way of processing might be weird,
    train_images = process(train_images)
    test_images = process(test_images)
    train_images = process(train_images)
    test_images = process(test_images)

    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    return (train_dataset,train_images, train_labels, test_images, test_labels)

def load_cifar10(args):
    """
        Loads the CIFAR10 dataset

        args (Namespace) Command line parameters from utils.cmd_input
    """
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    train_labels,test_labels = train_labels[:,0],test_labels[:,0] #because cifar labels are weird

    if str(args.anomaly_class) is not None:
        if args.anomaly_type == 'MISO':
            indicies = np.argwhere(train_labels == int(str(args.anomaly_class)))
            sample_indicies = random.sample(list(indicies[:,0]), int(args.percentage_anomaly*
                                                                      indicies.shape[0]))

            mask_train  = np.invert(train_labels == int(str(args.anomaly_class)))
        else: 
            indicies = np.argwhere(train_labels != int(str(args.anomaly_class)))
            sample_indicies = random.sample(list(indicies[:,0]), int(args.percentage_anomaly*
                                                                      indicies.shape[0]))

            mask_train  = train_labels == int(str(args.anomaly_class))

        mask_train[sample_indicies] = True
        train_images = train_images[mask_train]
        train_labels = train_labels[mask_train]

    if args.limit is not None:
        train_images = train_images[:args.limit,...]
        train_labels = train_labels[:args.limit,...]  
        test_images  = test_images[:args.limit,...]
        test_labels  = test_labels[:args.limit,...] 

    if args.patches:
        data  = get_patched_dataset(train_images,
                                    train_labels,
                                    test_images,
                                    test_labels,
                                    p_size = (1,args.patch_x, args.patch_y, 1),
                                    s_size = (1,args.patch_stride_x, args.patch_stride_y, 1))

        train_images, train_labels, test_images, test_labels = data
        train_images = process(train_images, per_image=False)
        test_images = process(test_images, per_image=False) # normalisation after patches results in misdirection.

    else: 
        train_images = process(train_images)
        test_images = process(test_images)

    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    return (train_dataset,train_images, train_labels, test_images, test_labels)

def load_mvtec(args):
    """
        Loads the MVTEC-AD dataset

        SIMO_class (str) is the SIMO class
        args.limit (int) sets a args.limit on the number of test and training samples
        args.percentage_anomaly (float) adds a percentage of the anomalous/novel class to the training set
    """
    (train_images, train_labels), (test_images, test_labels, test_masks) = get_mvtec_images(str(args.anomaly_class))

    
    if args.limit is not None:
        train_indx = np.random.permutation(len(train_images))[:args.limit]
        test_indx = np.random.permutation(len(test_images))[:args.limit]

        train_images = train_images[train_indx]
        train_labels = train_labels[train_indx]  
        test_images  = test_images[test_indx]
        test_labels  = test_labels[test_indx] 
        test_masks = test_masks[test_indx] 

    train_images = resize(train_images, (sizes[str(args.anomaly_class)], 
                                         sizes[str(args.anomaly_class)], 
                                         args.input_shape[-1]))

    test_images = resize(test_images, (sizes[str(args.anomaly_class)], 
                                       sizes[str(args.anomaly_class)], 
                                       args.input_shape[-1]))
    test_masks = np.expand_dims(test_masks,axis=-1)
    test_masks = resize(test_masks, (sizes[str(args.anomaly_class)], 
                                       sizes[str(args.anomaly_class)], 
                                       1))[...,0]
    if args.crop:
        cropped_images = random_crop(train_images,crop_size=(args.crop_x, args.crop_y))

    if args.patches:
        data  = get_patched_dataset(train_images,
                                    train_labels,
                                    test_images,
                                    test_labels,
                                    test_masks,
                                    p_size = (1,args.patch_x, args.patch_y, 1),
                                    s_size = (1,args.patch_stride_x, args.patch_stride_y, 1),
                                    central_crop=False)

        train_images, train_labels, test_images, test_labels, test_masks = data

        if args.crop:
            train_images = np.concatenate([train_images,cropped_images],axis=0)


    if args.rotate:
        train_images = random_rotation(train_images) 

    

    train_images = process(train_images, per_image=False)
    test_images =  process(test_images, per_image=False) # normalisation after patches results in misdirection.

    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    return (train_dataset,train_images, train_labels, test_images, test_labels, test_masks)
            
