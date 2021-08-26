import tensorflow as tf
import numpy as np
import os 
import pandas as pd
from sklearn.metrics import roc_curve, auc, average_precision_score, roc_auc_score
from math import isnan
from inference import infer, get_error
from utils import cmd_input 
from utils.data import reconstruct


def calculate_metrics(error,  
                      test_labels,
                      args):
    """
        Returns the AUROC and F1 score of a particular model 

        error (np.array): the reconstruction error of a given model
        test_labels (np.array): the test labels from the testing set
        args (Namespace):  arguments from utils.cmd_input

    """
    if args.anomaly_type == 'MISO':
        _auc = roc_auc_score(test_labels==args.anomaly_class, error)
    else:
        _auc = roc_auc_score(test_labels!=args.anomaly_class, error)

    return _auc


def get_classifcation(model_type,
                      model,
                      test_images,
                      test_labels,
                      args):
    """
        Returns the AUROC score of a particular model 

        model_type (str): type of model (AE,VAE,....)
        model (tf.keras.Model): the model used
        test_images (np.array): the test images from the testing set
        test_labels (np.array): the test labels from the testing set
        args (Namespace):  arguments from utils.cmd_input

    """
    x_hat = infer(model[0], test_images, args, 'DKNN')
    return -1

    if args.patches :
        error = get_error('AE', test_images, x_hat, mean=False)
        error, test_labels = reconstruct(error, args, test_labels)
        error =  error.mean(axis=tuple(range(1,error.ndim)))

    else:
        error = get_error('AE', test_images, x_hat, mean=True)

    auc = calculate_metrics(error,test_labels,args)
    return auc

def save_metrics(model_type,
                 args,
                 auc_reconstruction, 
                 seg_prc,
                 neighbour,
                 radius,
                 auc_latent,
                 seg_prc_nln,
                 seg_auc=None,
                 seg_auc_nln = None,
                 seg_iou=None,
                 seg_iou_nln = None,
                 dists_auc=None,
                 seg_dists_auc=None,
                 sum_auc=None,
                 mul_auc=None):
    
    """
        Either appends or saves a new .csv file with the top r and K 

        model_type (str): Type of model (AE,VAE,...)
        name (Namespace): args from utils/cmd_input
        auc_reconstruction (double): the AUROC reconstruction error 
        seg_prc (double): the F1 score for reconstruction error 
        neighbour (int): maximum number of neighbours (K) for best NLN error
        radius (double): radius size of best NLN error
        auc_latent (double): the AUROC score for NLN  
        seg_prc_nln (double): the F1 score for NLN
    """
    
    if isnan(radius): radius = 'nan'

    if not os.path.exists('outputs/results_{}_{}.csv'.format(args.data,
                                                             args.seed)):
        df = pd.DataFrame(columns = ['Model',
                                     'Name',
                                     'Latent_Dim',
                                     'Patch_Size',
                                     'Class',
                                     'Type',
                                     'Neighbour',
                                     #'Radius',
                                     'AUC_Reconstruction_Error',
                                     'AUC_NLN_Error',
                                     'Distance_AUC',
                                     'Sum_Recon_NLN_Dist',
                                     'Mul_Recon_NLN_Dist',
                                     'Seg_PRC',
                                     'Seg_PRC_NLN',
                                     'Segmentation_Reconstruction',
                                     'Segmentation_NLN',
                                     'Segmentation_Distance_AUC',
                                     'Segmentation_IOU',
                                     'Segmentation_IOU_NLN'])
    else:  
        df = pd.read_csv('outputs/results_{}_{}.csv'.format(args.data,
                                                            args.seed))


    df = df.append({'Model':model_type,
                    'Name':args.model_name,
                    'Latent_Dim':cmd_input.args.latent_dim,
                    'Patch_Size':args.patch_x,
                    'Class':args.anomaly_class,
                    'Type':args.anomaly_type,
                    'Neighbour':neighbour,
                    #'Radius':radius,
                    'AUC_Reconstruction_Error':auc_reconstruction,
                    'AUC_NLN_Error':auc_latent,
                    'Distance_AUC': dists_auc,
                    'Sum_Recon_NLN_Dist':sum_auc,
                    'Mul_Recon_NLN_Dist':mul_auc,
                    'Seg_PRC':seg_prc,
                    'Seg_PRC_NLN':seg_prc_nln,
                    'Segmentation_Reconstruction':seg_auc,
                    'Segmentation_NLN':seg_auc_nln,
                    'Segmentation_Distance_AUC':seg_dists_auc,
                    'Segmentation_IOU':seg_iou,
                    'Segmentation_IOU_NLN':seg_iou_nln},
                     ignore_index=True)

    df.to_csv('outputs/results_{}_{}.csv'.format(args.data,
                                                 args.seed),index=False)
