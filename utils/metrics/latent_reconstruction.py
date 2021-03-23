import tensorflow as tf
import numpy as np 
import pandas as pd
from time import time 
import copy
import pickle
from sklearn import metrics,neighbors
from model_loader import get_error, get_reconstructed
from joblib import Parallel, delayed
from utils.data import reconstruct, reconstruct_latent_patches
import itertools
import warnings
warnings.filterwarnings('ignore')


def get_sparse_errors(images,train_images, model,z,neighbours_idx,model_type,max_neigh):
    """
        This is different to the get_error function in model loader as it calculates ...
        the error of a single input based on all the decoded neighbours in z. 
        It works for the NLN algorithm with a changing number of neighbours.

        images (np.array): testing images
        train_images (np.array): images from training set 
        model (tf.keras.Model): the model used
        z (np.array): the embedding of the training set  
        neighbours_idx (np.array): the indexes of the neighbours of each point
        model_type (str): the type of model (AE,VAE,...)
        max_neigh (int): the maximum number of neightbours (K) if frNN
    """
    #TODO All of these methods are super inefficient
    error = []
    m = np.expand_dims(np.mean(images,axis=0),axis=0)

    if model_type == 'AE' or model_type == 'AAE' or model_type =='VAE': 

        with tf.device('/cpu:0'): # Because of weird tf memory management it is faster on cpu
            x_hat= model.decoder(z).numpy()

        for i,n in enumerate(neighbours_idx):
            if len(n) ==0: 
                temp = np.array([i])
                d = m 
            elif len(n) > max_neigh: 
                temp  = n[:max_neigh] 
                d = x_hat[temp.astype(int)]
            else: 
                temp  = n
                d = x_hat[temp.astype(int)]

            im = np.stack([images[i]]*temp.shape[-1],axis=0)
            error.append(np.mean(np.abs(d - im)**2))#,axis=0))

    elif model_type == 'DAE':
        with tf.device('/cpu:0'): # Because of weird tf memory management it is faster on cpu
            x_hat= model[0].decoder(z).numpy()

        for i,n in enumerate(neighbours_idx):
            if len(n) ==0: 
                temp = np.array([i])
                d = m
            elif len(n) > max_neigh: 
                temp  = n[:max_neigh] 
                d = x_hat[temp.astype(int)]
            else: 
                temp  = n
                d = x_hat[temp.astype(int)]

            im = np.stack([images[i]]*temp.shape[-1],axis=0)
            error.append(np.mean(np.abs(d - im)**2))

    elif model_type == 'DAE_disc':
        with tf.device('/cpu:0'): # Because of weird tf memory management it is faster on cpu
            x_hat= model[0].decoder(z).numpy()

            d_x_hat, _ = model[1](x_hat)
            d_x_hat_m, _ = model[1](m)

            d_x, _ = model[1](train_images)
            d_x_m, _ = model[1](m)

            d_x_hat = d_x_hat.numpy()
            d_x_hat_m = d_x_hat_m.numpy()

            d_x = d_x.numpy()
            d_x_m = d_x_m.numpy()

        for i,n in enumerate(neighbours_idx):
            if len(n) ==0: 
                temp = np.array([i])
                x_hats = m 
                d_x_hats = d_x_hat_m 
                d_xs = d_x_m

            elif len(n) > max_neigh: 
                temp  = n[:max_neigh] 
                x_hats = x_hat[temp.astype(int)]
                d_x_hats = d_x_hat[temp.astype(int)]
                d_xs = d_x[temp.astype(int)]

            else: 
                temp  = n
                x_hats = x_hat[temp.astype(int)]
                d_x_hats = d_x_hat[temp.astype(int)]
                d_xs = d_x[temp.astype(int)]


            ims = np.stack([images[i]]*temp.shape[-1],axis=0)
            reconstruction_error = np.mean(np.abs(x_hats - ims)**2)

            discriminator_error  = np.mean(np.abs(d_x_hats  - d_xs)**2)

            alpha = 0.9
            error.append((1-alpha)*reconstruction_error + alpha*discriminator_error)

    elif model_type == 'GANomaly': 
        with tf.device('/cpu:0'): # Because of weird tf memory management it is faster on cpu
            x_hat = model[0](images).numpy()
            z_hat = model[2](x_hat).numpy()
            z_hat_mean = model[2](m).numpy()

        for i,n in enumerate(neighbours_idx):
            if len(n) ==0: 
                temp = np.array([i])
                zs = z_hat_mean
            elif len(n) > max_neigh: 
                temp  = n[:max_neigh] 
                zs = z[temp.astype(int)]
            else: 
                temp  = n
                zs = z[temp.astype(int)]

            z_hats = np.stack([z_hat[i]]*temp.shape[-1],axis=0)
            error.append(np.mean(np.abs(zs - z_hats)**2))

    error = np.array(error)
    return error


def knn_nln(test_images,z_train, train_images_hat, neighbours_idx, model, model_type, args):
    """
        Gets the error for the KNN enabled NLN 

        images (np.array): testing images
        model (tf.keras.Model): the model used
        z (np.array): the embedding of the training set  
        neighbours_idx (np.array): the indexes of the neighbours of each point
        model_type (str): the type of model (AE,VAE,...)
        max_neigh (int): the maximum number of neightbours (K) if KNN 
    """
    if model_type != 'GANomaly': 
        test_images = np.stack([test_images]*neighbours_idx.shape[-1],axis=1) 
        neighbours = train_images_hat[neighbours_idx]
        error = np.mean(np.abs(test_images - neighbours), axis =1)

    if model_type == 'GANomaly': 
        with tf.device('/cpu:0'): # GPU cant handle the full training set :(
            z_test = model[0].encoder(test_images).numpy() 
            z_train_encoder = model[2](train_images_hat).numpy()

        z_test = np.stack([z_test]*neighbours_idx.shape[-1],axis=1) 
        neighbours = z_train_encoder[neighbours_idx]
        error = np.mean(np.abs(z_test- neighbours), axis =1)

    if args.patches:
        if model_type != 'GANomaly':
            error = reconstruct(error, args)
        else:
            error = reconstruct_latent_patches(error, args)

    error =  np.mean(error,axis=tuple(range(1,error.ndim)))
    return error

def nearest_error(model,
                 train_images,
                 test_images,
                 test_labels,
                 model_type,
                 args,
                 hera):
    """
        Calculated the NLN error for either frNN and KNN

        model (tf.keras.Model): the model used
        train_images (np.array): images from training set 
        test_images (np.array): testing images
        test_labels (np.array): labels of testing images
        model_type (str): the type of model (AE,VAE,...)
        args (Namespace): the argumenets from cmd_args
        hera (bool): if we use HERA

    """
    #if args.patches:## sample training_set for KNN
    #    inds = np.random.choice(np.arange(len(train_images)), 20000, replace=False)
    #    train_images = train_images[inds] 
        

    # inefficent, only need z
    z_query,error_query = get_error(model_type,model,test_images,return_z = True)
    with tf.device('/cpu:0'): # GPU cant handle the full training set :(
        z,_ = get_error(model_type,model,train_images,return_z = True)
        train_images_hat = get_reconstructed(model_type, model, train_images)

    d,max_auc,max_f1,max_neighbours,max_radius,index_counter = {},0,0,0,0,0
    z_query = z_query.numpy()
    z = z.numpy()
    
    #_,test_labels = reconstruct(test_images, args, test_labels)
    #(_, _), (_, test_labels) = tf.keras.datasets.cifar10.load_data()
    #test_labels = test_labels[:args.limit,0] #because cifar labels are weird

    for n_bour in args.neighbors:
        print('getting KNN of {}'.format(n_bour))
        if args.algorithm == 'knn':
            t = time()

            nbrs = neighbors.NearestNeighbors(n_neighbors=n_bour, 
                                              algorithm='ball_tree',
                                              n_jobs=-1).fit(z) # using KNN

            neighbours_idx =  nbrs.kneighbors(z_query,return_distance=False)#KNN

            error = knn_nln(test_images,
                            z, 
                            train_images_hat, 
                            neighbours_idx,
                            model,
                            model_type,
                            args)

            temp_args = [error,test_labels,args.anomaly_class,hera,args.neighbors,
                         [float('nan')],n_bour,float('nan'), max_auc,max_f1,max_neighbours,
                          max_radius,index_counter,d,t]

            (max_auc,max_f1,max_neighbours,
                    max_radius,index_counter,d) =  get_max_score(temp_args)



        elif args.algorithm == 'radius':
            for r in  args.radius:
                t = time()
                nbrs = neighbors.NearestNeighbors(radius=r, 
                                                  algorithm='ball_tree',
                                                  n_jobs=-1).fit(z) # using radius

                _,neighbours_idx =  nbrs.radius_neighbors(z_query,
                                                          return_distance=True,
                                                          sort_results=True)#radius

                error = get_sparse_errors(test_images,
                                          train_images,
                                          model,
                                          z,
                                          neighbours_idx,
                                          model_type,
                                          max_neigh=n_bour)
                temp_args = [error,test_labels,args.anomaly_class,hera,
                             args.neighbors, args.radius,n_bour,r, max_auc,
                             max_f1,max_neighbours,max_radius,index_counter,d,t]

                (max_auc,max_f1,max_neighbours,
                        max_radius,index_counter,d) = get_max_score(temp_args)

    with open('outputs/{}/{}/{}/latent_scores.pkl'.format(model_type,args.anomaly_class,args.model_name),'wb') as f:
        pickle.dump(d,f)

    return max_auc,max_f1,max_neighbours,max_radius

def get_max_score(args):
    """
        Find the maximum AUROC score for the model 

        args (Namespace): arguments from cmd_args
    """
    (error,test_labels,anomaly,hera,n_bours,
           radius,n_bour,rad,max_auc,max_f1,
           max_neighbours,max_radius,
           index_counter,d,t) = args

    if hera:
        mask = (np.char.find(test_labels, anomaly)>=0)
        fpr, tpr, thr  = metrics.roc_curve(mask,error)
        f1score = max([metrics.f1_score(y_true=mask,
                                        y_pred=error>t) for t in thr])

    else: 
        fpr, tpr, thr  = metrics.roc_curve(test_labels==anomaly,error)
        f1score = max([metrics.f1_score(y_true=test_labels==anomaly,
                                        y_pred=error>t) for t in thr])

    a_u_c = metrics.auc(fpr, tpr)

    if a_u_c > max_auc: max_auc = a_u_c; max_neighbours = n_bour;max_radius = rad;
    if f1score > max_f1: max_f1 = f1score 

    d[index_counter] = [n_bour,rad,a_u_c,f1score]
    index_counter+=1
    print("{}/{} f1-score: {}, auc-roc = {},  time elapsed = {}s".format(index_counter,
                                                         len(n_bours)*len(radius),
                                                         f1score,
                                                         a_u_c,
                                                         time()-t))

    return max_auc,max_f1,max_neighbours,max_radius,index_counter,d
