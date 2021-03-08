import tensorflow as tf
import numpy as np
import os 
import pandas as pd
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn import neighbors
from math import isnan
from model_loader import get_error
from utils import cmd_input 
from utils.metrics import get_sparse_errors


def accuracy_metrics(model,
                     train_images,
                     train_labels,
                     test_images,
                     test_labels,
                     model_type,
                     args):

    """
        Calculate accuracy metrics for MVTEC AD as reported by the paper

        model (tf.keras.Model): the model used
        train_images (np.array): non-anomalous images from train set 
        train_labels (np.array): non-anomalous labels of train set
        test_images (np.array): testing images
        test_labels (np.array): labels of testing images
        model_type (str): the type of model (AE,VAE,...)
        args (Namespace): the argumenets from cmd_args

    """
    # Get output from model #TODO: do we want to normalise?
    z_query, error  = get_error(model_type,model,test_images, return_z = True) 
#    error = normalise(e)
    get_acc(args.anomaly_class, test_labels, error)

    with tf.device('/cpu:0'): # Because of weird tf memory management it is faster on cpu
        z = model.encoder(train_images)

    nbrs = neighbors.NearestNeighbors(radius=0.5, 
                                      algorithm='ball_tree',
                                      n_jobs=-1).fit(z) # using radius

    _,neighbours_idx =  nbrs.radius_neighbors(z_query,
                                              return_distance=True,
                                              sort_results=True)#radius

    nln_error = get_sparse_errors(test_images, train_images, model, z, neighbours_idx,'AE', 2) 
    print('NLN')
    get_acc(args.anomaly_class,test_labels,nln_error)

    
    #x_hat_anomalous = get_reconstructed(model_type, model,anomalous_images)

def get_acc(anomaly_class, test_labels, error):
    # Find AUROC threshold that optimises max(TPR-FPR)
    print(anomaly_class)
    fpr, tpr, thr  = roc_curve(test_labels == anomaly_class, error)
    print('This is AUC {}'.format(auc(fpr, tpr)))

    thr = get_threshold(fpr,tpr,thr,'MD',test_labels, error,anomaly_class)
    print('threshold = {}'.format(thr))

    #print('FPR {}'.format(fpr))
    #print('TPR {}'.format(tpr))
    #print(thr)

    # Accuracy of detecting anomalies and non-anomalies using this threshold
    normal_accuracy = accuracy_score(test_labels == 'non_anomalous', error < thr)
    anomalous_accuracy = accuracy_score(test_labels == anomaly_class, error > thr)

    print('Anomalous Accuracy = {}'.format(anomalous_accuracy))
    print('Normal Accuracy = {}'.format(normal_accuracy))

    
def get_threshold(fpr,tpr,thr,flag,test_labels,error,anomaly_class):
    """
        Returns optimal threshold

        fpr (np.array): false positive rate
        tpr (np.array): true positive rate
        thr (np.array): thresholds for AUROC
    
    """
    if flag == 'MD':# MD = Maximise diff
        idx = np.argmax(tpr-fpr) 
    if flag == 'MA': # MA = Maximise average
        idx, temp = None, 0
        for i,t in enumerate(thr):
            normal_accuracy = accuracy_score(test_labels == 'non_anomalous', error < t)
            anomalous_accuracy = accuracy_score(test_labels == anomaly_class, error > t)
            m = np.mean([anomalous_accuracy, normal_accuracy])
            if  m > temp:
                idx = i
                temp = m
    return thr[idx]

def normalise(x):
    """
        Returns normalised input between 0 and 1

        x (np.array): 1D array to be Normalised
    """
    y = (x- np.min(x))/(np.max(x) - np.min(x))

    return y
