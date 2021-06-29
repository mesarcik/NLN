import tensorflow as tf
import numpy as np
from sklearn.metrics import (roc_curve,
                             auc, 
                             accuracy_score, 
                             average_precision_score, 
                             jaccard_score,
                             roc_auc_score, 
                             precision_recall_curve)
from inference import infer, get_error
from utils import cmd_input 
from utils.data import patches,sizes, reconstruct
from utils.metrics import nln, get_nln_errors
from reporting import plot_neighs

import time

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
    z = infer(model[0].encoder, train_images, args, 'encoder')
    z_query = infer(model[0].encoder, test_images, args, 'encoder')

    x_hat_train  = infer(model[0], train_images, args, 'AE')
    x_hat = infer(model[0], test_images, args, 'AE')

    error = get_error('AE', test_images, x_hat,mean=False)

    if args.patches:
        error_recon, labels_recon  = patches.reconstruct(error, args, test_labels) 
        masks_recon = patches.reconstruct(np.expand_dims(test_masks,axis=-1), args)[...,0] 
    else: 
        error_recon, labels_recon, masks_recon  = error, test_labels, test_masks 

    print('Original With Reconstruction')
    error_agg =  np.mean(error_recon ,axis=tuple(range(1,error_recon.ndim)))
    cl_auc , normal_accuracy, anomalous_accuracy = get_acc(args.anomaly_class, labels_recon, error_agg)
    
    seg_auc, seg_prc = get_segmentation(error_recon, masks_recon, labels_recon, args)
    seg_iou= -1#iou_score(error_recon, masks_recon)


    seg_auc_nlns, seg_prc_nlns, dist_aucs, seg_aucs_dist, seg_iou_nlns = [], [], [], [],[]
    print('NLN With Reconstruction')
    for n in args.neighbors:
        neighbours_dist, neighbours_idx, x_hat_train, neighbour_mask =  nln(z, 
                                                                            z_query, 
                                                                            x_hat_train, 
                                                                            args.algorithm, 
                                                                            n,
                                                                            -1)
        nln_error = get_nln_errors(model,
                                   'AE',
                                   z_query,
                                   z,
                                   test_images,
                                   x_hat_train,
                                   neighbours_idx,
                                   neighbour_mask,
                                   args)


        if args.patches:
            if nln_error.ndim ==4:
                nln_error_recon = patches.reconstruct(nln_error, args)
            else:
                nln_error_recon = patches.reconstruct_latent_patches(nln_error, args)
        else: nln_error_recon = nln_error
        

        error_agg =  np.mean(nln_error_recon ,axis=tuple(range(1,nln_error_recon.ndim)))
        cl_auc_nln , normal_accuracy_nln, anomalous_accuracy_nln = get_acc(args.anomaly_class,labels_recon, error_agg)
        iou_nln = -1#iou_score(nln_error_recon, masks_recon)
        seg_iou_nlns.append(iou_nln)

        seg_auc_nln,seg_prc_nln = get_segmentation(nln_error_recon, masks_recon, labels_recon, args)
        seg_auc_nlns.append(seg_auc_nln)
        seg_prc_nlns.append(seg_prc_nln)

        dists_recon = get_dists(neighbours_dist, args)

        dists = np.max(dists_recon, axis = tuple(range(1,dists_recon.ndim)))
        dists= roc_auc_score(labels_recon== args.anomaly_class, dists) 
        dists_seg,_ = get_segmentation(dists_recon, masks_recon, labels_recon, args)
#        dists_auc = roc_auc_score(test_masks.flatten()>0, np.mean(error,axis=-1).flatten())
        dist_aucs.append(dists)
        seg_aucs_dist.append(dists_seg)

#        fpr, tpr, thr  = roc_curve(labels_recon==args.anomaly_class, dists)
        print('\nDists AUC = {}\n'.format(dists))


    seg_auc_nln = max(seg_auc_nlns)
    seg_prc_nln = max(seg_prc_nlns)
    dists_auc = max(dist_aucs)
    seg_dists_auc = max(seg_aucs_dist)
    seg_iou_nln = max(seg_iou_nlns)

    print('Max seg_auc neighbor= {}\nMax seg_prc neighbor={}\nMax dist_uac neighbor ={}\nMax seg_iou neigh={}'.format(
                                                                                    args.neighbors[np.argmax(seg_auc_nlns)],
                                                                                    args.neighbors[np.argmax(seg_prc_nlns)],
                                                                                    args.neighbors[np.argmax(dist_aucs)],
                                                                                    args.neighbors[np.argmax(seg_aucs_dist)],
                                                                                    args.neighbors[np.argmax(seg_iou_nlns)],
                                                                                    ))
    with open("outputs/neighbour_results.csv", "a") as myfile:
        myfile.write('{},{},{},{},{},{},{}\n'.format(model_type, 
                                                     args.anomaly_class, 
                                                     round(seg_auc_nln,3), 
                                                     args.neighbors[np.argmax(seg_auc_nlns)],
                                                     round(seg_prc_nln,3),
                                                     args.neighbors[np.argmax(seg_prc_nlns)],
                                                     round(dists_auc,3),
                                                     args.neighbors[np.argmax(dist_aucs)],
                                                     round(seg_dists_auc,3),
                                                     args.neighbors[np.argmax(dist_aucs)],
                                                     round(seg_iou_nln,3),
                                                     args.neighbors[np.argmax(seg_iou_nlns)],
                                                     ))

    plot_neighs(test_images, test_labels, test_masks, x_hat, x_hat_train[neighbours_idx], neighbours_dist, model_type, args)
    
    #x_hat_anomalous = get_reconstructed(model_type, model,anomalous_images)
    return seg_auc, seg_auc_nln, dists_auc, seg_dists_auc, seg_prc, seg_prc_nln, seg_iou, seg_iou_nln

def get_segmentation(error, test_masks, test_labels, args):
    """
        Calculates AUROC result of segmentation
    """
    fpr, tpr, thr  = roc_curve(test_masks.flatten()>0, np.max(error,axis=-1).flatten())
#    prc = average_precision_score(test_masks.flatten()>0, np.mean(error,axis=-1).flatten())
    precision, recall, thresholds = precision_recall_curve(test_masks.flatten()>0, np.max(error,axis=-1).flatten())
    prc = auc(recall, precision)
    AUC= roc_auc_score(test_masks.flatten()>0, np.mean(error,axis=-1).flatten())#, max_fpr=0.3)

    print('This is Segementation AUC {}'.format(AUC))
    return AUC,prc

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
#    AUC = auc(fpr,tpr)
#    AUC = average_precision_score(test_labels == anomaly_class, error) 
    AUC= roc_auc_score(test_labels==anomaly_class,error)
    print('This is AUC {}'.format(AUC))

    thr = get_threshold(fpr,tpr,thr,'MD',test_labels, error,anomaly_class)
    print('threshold = {}'.format(thr))

    # Accuracy of detecting anomalies and non-anomalies using this threshold
    normal_accuracy = accuracy_score(test_labels == 'non_anomalous', error < thr)
    anomalous_accuracy = accuracy_score(test_labels == anomaly_class, error > thr)

    print('Anomalous Accuracy = {}'.format(anomalous_accuracy))
    print('Normal Accuracy = {}'.format(normal_accuracy))

    return AUC, normal_accuracy, anomalous_accuracy

    
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

def get_dists(neighbours_dist, args):

    dists = np.mean(neighbours_dist, axis = tuple(range(1,neighbours_dist.ndim)))
    if args.patches:
        dists = np.array([[d]*args.patch_x**2 for i,d in enumerate(dists)]).reshape(len(dists), args.patch_x, args.patch_y)
        dists_recon = reconstruct(np.expand_dims(dists,axis=-1),args)
        return dists_recon
    else:
        return dists 

def iou_score(error, test_masks):
    fpr,tpr, thr = roc_curve(test_masks.flatten()>0, np.mean(error,axis=-1).flatten())
    idx = np.argmax(tpr-fpr) 
    threshold = thr[idx]
    #mask = fpr<=0.3

    #thr = thr[mask]
    #fpr = fpr[mask] 
    #tpr = tpr[mask]

    pro, iou = [], []

#    for threshold in np.linspace(np.min(thr), np.max(thr),10):
    thresholded =np.mean(error,axis=-1) >=threshold
    #result = np.where(test_masks.flatten() == thresholded.flatten(),
    #                  test_masks.flatten(),
    #                  0)
    #val, counts = np.unique(result, return_counts=True)
    #pro.append(counts[-1]/np.sum(counts))
    t = time.time()
    iou.append(jaccard_score(test_masks.flatten()>0, thresholded.flatten()))
    print('Time elapsed for jaccard score {}'.format(time.time() -t))

    return max(iou) #auc(fpr,tpr), auc(fpr,pro), auc(fpr,iou) 

