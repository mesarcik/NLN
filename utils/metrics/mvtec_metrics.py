import tensorflow as tf
import numpy as np
import os 
import pandas as pd
from sklearn.metrics import roc_curve, auc, accuracy_score
from math import isnan
from model_loader import get_error
from utils import cmd_input 


def accuracy_metrics(model,
                     test_images,
                     test_labels,
                     model_type,
                     args):

    """
        Calculate accuracy metrics for MVTEC AD as reported by the paper

        model (tf.keras.Model): the model used
        normal_images (np.array): non-anomalous images from test set 
        normal_labels (np.array): non-anomalous labels of test set
        anomalous_images (np.array): testing images
        anomalous_labels (np.array): labels of testing images
        model_type (str): the type of model (AE,VAE,...)
        args (Namespace): the argumenets from cmd_args

    """
    # Get output from model #TODO: do we want to normalise?
    error = normalise(get_error(model_type,model,test_images)) 
    #print(error)
    
    #x_hat_anomalous = get_reconstructed(model_type, model,anomalous_images)

    # Find AUROC threshold that optimises max(TPR-FPR)
    print(args.anomaly_class)
    fpr, tpr, thr  = roc_curve(test_labels == args.anomaly_class, error)
    print('This is AUC {}'.format(auc(fpr, tpr)))

    thr = get_threshold(fpr,tpr,thr,'MA',test_labels, error,args)

    #print('FPR {}'.format(fpr))
    #print('TPR {}'.format(tpr))
    #print(thr)

    # Accuracy of detecting anomalies and non-anomalies using this threshold
    normal_accuracy = accuracy_score(test_labels == 'non_anomalous', error < thr)
    anomalous_accuracy = accuracy_score(test_labels == args.anomaly_class, error > thr)

    print('Anomalous Accuracy = {}'.format(anomalous_accuracy))
    print('Normal Accuracy = {}'.format(normal_accuracy))

    return thr

    
def get_threshold(fpr,tpr,thr,flag,test_labels,error,args):
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
            anomalous_accuracy = accuracy_score(test_labels == args.anomaly_class, error > t)
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
