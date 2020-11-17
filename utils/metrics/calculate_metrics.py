import tensorflow as tf
import numpy as np
import os 
import pandas as pd
from sklearn.metrics import roc_curve, auc, f1_score
from math import isnan
from model_loader import get_error
from utils import cmd_input 


def calculate_metrics(error,  
                      test_labels,
                      anomaly,
                      hera=False,
                      f1=False):

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
                      f1=False):

    error = get_error(model_type,model,test_images)
    auc,f1 = calculate_metrics(error,test_labels,anomaly,hera,f1)
    return auc,f1

def save_metrics(model_type,
                 name,
                 dataset,
                 anomaly,
                 auc_reconstruction, 
                 f1_reconstruction,
                 neighbour,
                 radius,
                 auc_latent,
                 f1_latent):
    
    if isnan(radius): radius = 'nan'

    if not os.path.exists('outputs/results_{}.csv'.format(dataset)):
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
        df = pd.read_csv('outputs/results_{}.csv'.format(dataset))


    df = df.append({'Model':model_type,
                    'Name':name,
                    'Latent_Dim':cmd_input.args.latent_dim,
                    'Class':anomaly,
                    'Neighbour':neighbour,
                    'Radius':radius,
                    'AUC_Reconstruction_Error':auc_reconstruction,
                    'AUC_Latent_Error':auc_latent,
                    'F1_Reconstruction_Error':f1_reconstruction,
                    'F1_Latent_Error':f1_latent},
                     ignore_index=True)

    df.to_csv('outputs/results_{}.csv'.format(dataset),index=False)
