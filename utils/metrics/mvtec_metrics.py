import tensorflow as tf
import numpy as np
import os 
import pandas as pd
from sklearn.metrics import roc_curve, auc, accuracy_score
from math import isnan
from model_loader import get_error
from utils import cmd_input 


def accuracy_metrics(model,
                     normal_images,
                     normal_labels,
                     anomalous_images,
                     anomalous_labels,
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
    error_normal = normalise(get_error(model_type,model,normal_images)) 
    print(error_normal)

    error_anomalous = normalise(get_error(model_type,model,anomalous_images))
    print(error_anomalous)
    #x_hat_anomalous = get_reconstructed(model_type, model,anomalous_images)

    # Find AUROC threshold that optimises max(TPR-FPR)
    fpr, tpr, thr  = roc_curve(normal_labels, error_normal)
    thr_normal = get_threshold(fpr,tpr,thr)
    print(thr_normal)

    fpr, tpr, thr  = roc_curve(anomalous_labels, error_anomalous)
    thr_anomalous = get_threshold(fpr,tpr,thr)
    print(fpr)
    print(tpr)
    print(thr_anomalous)

    # Accuracy of detecting anomalies and non-anomalies using this threshold
    normal_accuracy = accuracy_score(normal_labels, error_normal> thr_anomalous)
    anomalous_accuracy = accuracy_score(anomalous_labels, error_anomalous> thr_anomalous)

    print('Anomalous Accuracy = {}'.format(anomalous_accuracy))
    print('Normal Accuracy = {}'.format(normal_accuracy))

    
def get_threshold(fpr,tpr,thr):
    """
        Returns optimal threshold

        fpr (np.array): false positive rate
        tpr (np.array): true positive rate
        thr (np.array): thresholds for AUROC
    
    """
    idx = np.argmax(tpr-fpr) 
    return thr[idx]

def normalise(x):
    """
        Returns normalised input between 0 and 1

        x (np.array): 1D array to be Normalised
    """
    y = (x- np.min(x))/(np.max(x) - np.min(x))

    return y
