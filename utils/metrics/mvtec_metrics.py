import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score
from inference import infer, get_error
from utils import cmd_input 
from utils.data import patches 
from utils.metrics import nln, get_nln_errors
from reporting import plot_neighs



def accuracy_metrics(model,
                     train_images,
                     test_images,
                     test_labels,
                     test_masks,
                     model_type,
                     max_neighbours,
                     max_radius,
                     args):

    """
        Calculate accuracy metrics for MVTEC AD as reported by the paper


        Parameters
        ----------
        model (tf.keras.Model): the model used
        train_images (np.array): non-anomalous images from train set 
        test_images (np.array): testing images
        test_labels (np.array): labels of testing images
        test_masks (np.array): ground truth masks for the testing images
        model_type (str): the type of model (AE,VAE,...)
        args (Namespace): the argumenets from cmd_args
        max_neighbours (int): number of neighbours resulting in best AUROC
        max_radius (double): size of radius resulting in best AUROC

        Returns
        -------

    """
    # Get output from model #TODO: do we want to normalise?
    
    x_hat = infer(model[0], test_images, args, 'AE')
    error = get_error('AE', test_images, x_hat,mean=False)

    error_recon, labels_recon  = patches.reconstruct(error, args, test_labels) 
    masks_recon = patches.reconstruct(np.expand_dims(test_masks,axis=-1), args)[...,0] 

    print('Original With Reconstruction')
    error_agg =  np.mean(error_recon ,axis=tuple(range(1,error_recon.ndim)))
    cl_auc , normal_accuracy, anomalous_accuracy = get_acc(args.anomaly_class, labels_recon, error_agg)
    
    seg_auc = get_segmentation(error_recon, masks_recon, labels_recon, args)

    with open("outputs/test_results.csv", "a") as myfile:
        myfile.write('{},{},{},{},{},{},{}\n'.format(model_type, 
                                                         args.anomaly_class, 
                                                         "False", 
                                                         seg_auc, 
                                                         cl_auc, 
                                                         normal_accuracy, 
                                                         anomalous_accuracy))

    z = infer(model[0].encoder, train_images, args, 'encoder')
    z_query = infer(model[0].encoder, test_images, args, 'encoder')

    x_hat_train  = infer(model[0], train_images, args, 'AE')
    x_hat = infer(model[0], test_images, args, 'AE')

    neighbours_dist, neighbours_idx, x_hat_train, neighbour_mask =  nln(z, 
                                                                        z_query, 
                                                                        x_hat_train, 
                                                                        args.algorithm, 
                                                                        max_neighbours,
                                                                        max_radius)
    nln_error = get_nln_errors(model,
                               'AE',
                               z_query,
                               z,
                               test_images,
                               x_hat_train,
                               neighbours_idx,
                               neighbour_mask,
                               args)

    if nln_error.ndim ==4:
        nln_error_recon = patches.reconstruct(nln_error, args)
    else:
        nln_error_recon = patches.reconstruct_latent_patches(nln_error, args)

    print('NLN With Reconstruction')
    error_agg =  np.mean(nln_error_recon ,axis=tuple(range(1,nln_error_recon.ndim)))
    cl_auc_nln , normal_accuracy_nln, anomalous_accuracy_nln = get_acc(args.anomaly_class,labels_recon, error_agg)
    seg_auc_nln = get_segmentation(nln_error_recon, masks_recon, labels_recon, args)


    with open("outputs/test_results.csv", "a") as myfile:
        myfile.write('{},{},{},{},{},{},{}\n'.format(model_type, 
                                                         args.anomaly_class, 
                                                         "True", 
                                                         seg_auc_nln, 
                                                         cl_auc_nln, 
                                                         normal_accuracy_nln, 
                                                         anomalous_accuracy_nln))

    plot_neighs(test_images, test_labels, test_masks, x_hat, x_hat_train[neighbours_idx], neighbours_dist, args)
    
    #x_hat_anomalous = get_reconstructed(model_type, model,anomalous_images)

def get_segmentation(error, test_masks, test_labels, args):
    """
        Calculates AUROC result of segmentation
    """
    fpr, tpr, thr  = roc_curve(test_masks.flatten()>0, np.mean(error,axis=-1).flatten())
    print('This is Segementation AUC {}'.format(auc(fpr, tpr)))
    return auc(fpr,tpr)

    #thr = get_threshold(fpr,tpr,thr,'MD',test_labels, error.mean(axis=tuple(range(1,error.ndim))),args.anomaly_class)

    #error = error.mean(axis=tuple(range(1,error.ndim)))
    #normal_accuracy = accuracy_score(test_labels == 'non_anomalous', error < thr)
    #anomalous_accuracy = accuracy_score(test_labels == args.anomaly_class, error > thr)

    #print('Segmenation based Anomalous Accuracy = {}'.format(anomalous_accuracy))
    #print('Segmentation based Normal Accuracy = {}'.format(normal_accuracy))
    

def get_acc(anomaly_class, test_labels, error):
    # Find AUROC threshold that optimises max(TPR-FPR)
    print(anomaly_class)
    fpr, tpr, thr  = roc_curve(test_labels == anomaly_class, error)
    print('This is AUC {}'.format(auc(fpr, tpr)))

    thr = get_threshold(fpr,tpr,thr,'MD',test_labels, error,anomaly_class)
    print('threshold = {}'.format(thr))

    # Accuracy of detecting anomalies and non-anomalies using this threshold
    normal_accuracy = accuracy_score(test_labels == 'non_anomalous', error < thr)
    anomalous_accuracy = accuracy_score(test_labels == anomaly_class, error > thr)

    print('Anomalous Accuracy = {}'.format(anomalous_accuracy))
    print('Normal Accuracy = {}'.format(normal_accuracy))

    return auc(fpr,tpr), normal_accuracy, anomalous_accuracy

    
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
