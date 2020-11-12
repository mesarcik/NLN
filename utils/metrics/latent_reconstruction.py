import tensorflow as tf
import numpy as np 
import pandas as pd
from time import time 
import copy
import pickle
from sklearn import metrics,neighbors
from model_loader import get_model, get_error
from joblib import Parallel, delayed
import itertools
import warnings
warnings.filterwarnings('ignore')


def get_sparse_errors(images,train_images, model,z,neighbours_idx,model_type,max_neigh):
    """
        This is different to the get_error function in model loader as it calculates ...
        the error of a single input based on all the decoded neighbours in z
    """
    #TODO All of these methods are super inefficient
    error = []
    if model_type == 'AE' or model_type == 'AAE' or model_type =='VAE': 

        with tf.device('/cpu:0'): # Because of weird tf memory management it is faster on cpu
            x_hat= model.decoder(z).numpy()

        for i,n in enumerate(neighbours_idx):
            if len(n) ==0: temp = np.array([i])
            elif len(n) > max_neigh: temp  = n[:max_neigh] 
            else: temp  = n

            d = x_hat[temp.astype(int)]
            im = np.stack([images[i]]*temp.shape[-1],axis=0)
            error.append(np.mean(np.abs(d - im)**2))

    elif model_type == 'DAE':
        with tf.device('/cpu:0'): # Because of weird tf memory management it is faster on cpu
            x_hat= model[0].decoder(z).numpy()

        for i,n in enumerate(neighbours_idx):
            if len(n) ==0: temp = np.array([i])
            elif len(n) > max_neigh: temp  = n[:max_neigh] 
            else: temp  = n

            d = x_hat[temp.astype(int)]
            im = np.stack([images[i]]*temp.shape[-1],axis=0)
            error.append(np.mean(np.abs(d - im)**2))

    elif model_type == 'DAE_disc':
        with tf.device('/cpu:0'): # Because of weird tf memory management it is faster on cpu
            x_hat= model[0].decoder(z).numpy()
            d_x_hat, _ = model[1](x_hat)
            d_x, _ = model[1](train_images)

            d_x_hat = d_x_hat.numpy()
            d_x = d_x.numpy()

        for i,n in enumerate(neighbours_idx):
            if len(n) ==0: temp = np.array([i])
            elif len(n) > max_neigh: temp  = n[:max_neigh] 
            else: temp  = n

            x_hats = x_hat[temp.astype(int)]
            ims = np.stack([images[i]]*temp.shape[-1],axis=0)
            reconstruction_error = np.mean(np.abs(x_hats - ims)**2)

            d_x_hats = d_x_hat[temp.astype(int)]
            d_xs = d_x[temp.astype(int)]
            discriminator_error  = np.mean(np.abs(d_x_hats  - d_xs)**2)

            alpha = 0.9
            error.append((1-alpha)*reconstruction_error + alpha*discriminator_error)

    elif model_type == 'GANomaly': 
        with tf.device('/cpu:0'): # Because of weird tf memory management it is faster on cpu
            x_hat = model[0](images).numpy()
            z_hat = model[2](x_hat).numpy()

        for i,n in enumerate(neighbours_idx):
            if len(n) ==0: temp = np.array([i])
            elif len(n) > max_neigh: temp  = n[:max_neigh] 
            else: temp  = n

            zs = z[temp.astype(int)]
            z_hats = np.stack([z_hat[i]]*temp.shape[-1],axis=0)
            error.append(np.mean(np.abs(zs - z_hats)**2))

    error = np.array(error)
    return error

def get_padded_output(neighbours_idx,decoded,max_neigh):
    """
        Takes in the neighbours list and returns the associated padding tuple for np.pad 
        This is done in order to make the np.array square
    """
    # note this function is depreciated
    padded_output = [] 

    for i,n in enumerate(neighbours_idx):
        if len(n) ==0: temp = np.array([i])
        elif len(n) > max_neigh: temp  = n[:max_neigh] 
        else: temp  = n

        npad = ((0,max_neigh- len(temp)),(0,0),(0,0),(0,0))
        # if there are no neighbours, then replace with the orignal porint
        padded_output.append(np.pad(decoded[temp],
                             pad_width=npad,
                             mode='constant',
                             constant_values=(np.nan,)))
        

    return np.array(padded_output)


def get_n_errors(images, model,z,neighbours_idx,model_type,max_neigh=None):
    """
        This is different to the get_error function in model loader as it calculates ...
        the error of a single input based on all the decoded neighbours in z
    """
    if max_neigh is None:
        max_neigh =  max([sublist.shape[0] for sublist in neighbours_idx])
    images = np.stack([images]*max_neigh,axis=1) 
    # images of shape (n,32,32,1)->(n,d,32,32,1)

    if model_type == 'AE' or model_type == 'AAE' or model_type =='VAE':
        x_hat = np.array([model.decoder(z[n]) for n in neighbours_idx])
        error = np.abs(images - x_hat)**2

    if model_type == 'DAE':
        x_hat = np.array([model[0].decoder(z[n]) for n in neighbours_idx])
        error = np.abs(images - x_hat)**2

    elif model_type == 'GANomaly':
        zs = np.array([z[n] for n in neighbours_idx])
        x_hats = np.array([model[0](image) for image in images])
        z_hats = np.array([model[2](x_hat) for x_hat in x_hats])
        error = tf.square(tf.subtract(z_hats, zs)).numpy()

    elif model_type == 'BIGAN':
        # Note this is not using the discriminator error
        x_hat = np.array([model[1].decoder(z[n]) for n in neighbours_idx])
        error = tf.square(tf.subtract(images, x_hat)).numpy()

    elif model_type == 'GPND' or model_type == 'VAEGAN':
        # Note this is not using the discriminator error
        x_hat = np.array([model[0].decoder(z[n]) for n in neighbours_idx])
        error = tf.square(tf.subtract(images, x_hat)).numpy()

    error =  np.nanmean(error,axis=tuple(range(1,error.ndim)))
    return error

def nearest_error(model,
                 train_images,
                 test_images,
                 test_labels,
                 model_type,
                 args,
                 hera):

    # inefficent, only need z
    z_query,error_query = get_error(model_type,model,test_images,return_z = True)
    with tf.device('/cpu:0'): # GPU cant handle the full training set :(
        z,_ = get_error(model_type,model,train_images,return_z = True)

    d,max_auc,max_f1,max_neighbours,max_radius,index_counter = {},0,0,0,0,0
    z_query = z_query.numpy()
    z = z.numpy()
    
    for n_bour in args.neighbors:
        if args.algorithm == 'knn':
            t = time()

            nbrs = neighbors.NearestNeighbors(n_neighbors=n_bour, 
                                              algorithm='ball_tree',
                                              n_jobs=-1).fit(z) # using KNN

            neighbours_idx =  nbrs.kneighbors(z_query,return_distance=False)#KNN
            error = get_n_errors(test_images,model,z,neighbours_idx,model_type)
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
