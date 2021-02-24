import pickle
import numpy as np
import os
from tqdm import tqdm 
from imageio import imread
from glob import glob

def get_mvtec_images(SIMO_class, directory='datasets/MVTecAD/'):
    """"
        Walks through MVTEC dataset and returns data in the same structure as tf
        
        SIMO_class (str): Anomalous class 
        directory (str): Directory where MVTecAD dataset resides
    """

    train_images, test_images, train_labels ,test_labels, test_mask  = [], [], [], [], []
    
    # if the training dataset has already been created then return that
    if os.path.exists('{}{}.pickle'.format(directory,SIMO_class)):
        print('{}{}.pickle loading'.format(directory, SIMO_class))
        with open('{}/{}.pickle'.format(directory,SIMO_class),'rb') as f:
            return pickle.load(f)

    print('Creating data for {}'.format(SIMO_class))
    for f in tqdm(glob("{}/{}/train/good/*.png".format(directory,SIMO_class))): 
        img = imread(f)
        train_images.append(img) 
        train_labels.append('non_anomalous')

    for f in tqdm(glob("{}/{}/test/*/*.png".format(directory,SIMO_class))): 
        img = imread(f)
        test_images.append(img)
        if 'good' in f: 
            test_labels.append('non_anomalous')
            test_mask.append(np.zeros([img.shape[0],
                                       img.shape[1]]))
        else:
            test_labels.append(SIMO_class)
            f_ = f.split('/')
            img_mask = imread(os.path.join(f_[0], 
                                           f_[1], 
                                           f_[2], 
                                           f_[3], 
                                           'ground_truth', 
                                           f_[5], 
                                           f_[6].split('.')[0] + '_mask.png'))
            test_mask.append(img_mask)

    pickle.dump(((np.array(train_images), 
                  np.array(train_labels)),
                (np.array(test_images), 
                 np.array(test_labels), 
                 np.array(test_mask))),
                open('{}/{}.pickle'.format(directory,SIMO_class), 'wb'), protocol=1)

    return (np.array(train_images), np.array(train_labels)), (np.array(test_images), np.array(test_labels), np.array(test_mask))
