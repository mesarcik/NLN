import tensorflow as tf
import numpy as np
import os 
import pandas as pd
from sklearn.metrics import roc_curve, auc, f1_score
from math import isnan
from model_loader import get_error
from utils import cmd_input 
from utils.data import reconstruct


def calculate_metrics(error,  
                      test_labels,
                      anomaly,
                      hera=False,
                      f1=False):
    """
        Returns the AUROC and F1 score of a particular model 

        error (np.array): the reconstruction error of a given model
        test_labels (np.array): the test labels from the testing set
        anomaly (int): the anomlous class label
        hera (bool): if the HERA training set is used
        f1 (bool): return f1 score?

    """
    error = (error - np.min(error))/(np.max(error) - np.min(error))
    if hera:
        mask= (np.char.find(test_labels, anomaly)>=0)
        fpr, tpr, thr  = roc_curve(mask, error)
        if f1:
            f1 = max([f1_score(mask,error>t) for t in thr])
        else:
            f1 =  None
    else:
        fpr, tpr, thr  = roc_curve(test_labels==anomaly, error)
        if f1:
            f1 = max([f1_score(test_labels==anomaly,error>t) for t in thr])
        else: 
            f1 = None

    return auc(fpr, tpr),f1


def get_classifcation(model_type,
                      model,
                      test_images,
                      test_labels,
                      anomaly,
                      hera=False,
                      f1=False,
                      args=None):
    """
        Returns the AUROC and F1 score of a particular model 

        model_type (str): type of model (AE,VAE,....)
        model (tf.keras.Model): the model used
        test_images (np.array): the test images from the testing set
        test_labels (np.array): the test labels from the testing set
        anomaly (int): the anomlous class label
        hera (bool): if the HERA training set is used
        f1 (bool): return f1 score?

    """

    if args is None:
        error = get_error(model_type,model,test_images)

    else:
        with tf.device('/cpu:0'): 
            model_output = model[0](test_images)
            z = model[0].encoder(test_images)
            error = np.abs(test_images -  model_output.numpy())
            error, _ = reconstruct(error, args, test_labels)
            error =  np.mean(error,axis=tuple(range(1,error.ndim)))

        _,test_labels = reconstruct(test_images, args, test_labels)
        (_, _), (_, test_labels) = tf.keras.datasets.cifar10.load_data()
        test_labels = test_labels[:args.limit,0] #because cifar labels are weird

    auc,f1 = calculate_metrics(error,test_labels,anomaly,hera,f1)
    return auc,f1

def save_metrics(model_type,
                 args,
                 auc_reconstruction, 
                 f1_reconstruction,
                 neighbour,
                 radius,
                 auc_latent,
                 f1_latent):
    """
        Either appends or saves a new .csv file with the top r and K 

        model_type (str): Type of model (AE,VAE,...)
        name (Namespace): args from utils/cmd_input
        auc_reconstruction (double): the AUROC reconstruction error 
        f1_reconstruction (double): the F1 score for reconstruction error 
        neighbour (int): maximum number of neighbours (K) for best NLN error
        radius (double): radius size of best NLN error
        auc_latent (double): the AUROC score for NLN  
        f1_latent (double): the F1 score for NLN
    """
    
    if isnan(radius): radius = 'nan'

    if not os.path.exists('outputs/results_{}_{}.csv'.format(args.data,
                                                             args.seed)):
        df = pd.DataFrame(columns = ['Model',
                                     'Name',
                                     'Latent_Dim',
                                     'Class',
                                     'Neighbour',
                                     'Radius',
                                     'AUC_Reconstruction_Error',
                                     'AUC_Latent_Error',
                                     'F1_Reconstruction_Error',
                                     'F1_Latent_Error'])
    else:  
        df = pd.read_csv('outputs/results_{}_{}.csv'.format(args.data,
                                                            args.seed))


    df = df.append({'Model':model_type,
                    'Name':args.model_name,
                    'Latent_Dim':cmd_input.args.latent_dim,
                    'Class':args.anomaly_class,
                    'Neighbour':neighbour,
                    'Radius':radius,
                    'AUC_Reconstruction_Error':auc_reconstruction,
                    'AUC_Latent_Error':auc_latent,
                    'F1_Reconstruction_Error':f1_reconstruction,
                    'F1_Latent_Error':f1_latent},
                     ignore_index=True)

    df.to_csv('outputs/results_{}_{}.csv'.format(args.data,
                                                 args.seed),index=False)
