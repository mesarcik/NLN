import tensorflow as tf
import numpy as np 
from time import time 
import pickle
from sklearn import metrics,neighbors
from inference import infer, get_error
from utils.data import reconstruct, reconstruct_latent_patches
from model_config import *
import itertools
import warnings
warnings.filterwarnings('ignore')



def nln(z, z_query, x_hat_train, algorithm, neighbours, radius=None):
    """
        Calculates the nearest neighbours using either frNN or KNN 

        Parameters
        ----------
        z (np.array): training set latent space
        z_query (np.array): test set latent space 
        x_hat_train (np.array): reconstruction of training data
        algorithm (str): KNN or frNN
        neighbours (int): number of neighbours 
        radius (double): the frnn radius
        
        Returns
        -------
        (np.array, np.array, np.array, np.array)

    """
    if algorithm == 'knn':
        nbrs = neighbors.NearestNeighbors(n_neighbors= neighbours,
                                          algorithm='ball_tree',
                                          n_jobs=-1).fit(z) 

        neighbours_dist, neighbours_idx =  nbrs.kneighbors(z_query,return_distance=True)#KNN
        neighbour_mask  = np.zeros([len(neighbours_idx)], dtype=bool)

    elif algorithm == 'frnn':
        nbrs = neighbors.NearestNeighbors(radius=radius, 
                                          algorithm='ball_tree',
                                          n_jobs=-1).fit(z) # using radius

        neighbours_dist, neighbours_idx =  nbrs.radius_neighbors(z_query,
                                                  return_distance=True,
                                                  sort_results=True)#radius
        neighbours_idx_ = -1*np.ones([len(neighbours_idx), neighbours],dtype=int)
        neighbour_mask  = np.zeros([len(neighbours_idx)], dtype=bool)

        for i,n in enumerate(neighbours_idx):
            if len(n) == 0:
                neighbour_mask[i] = [True]
                pass
            elif len(n) > neighbours:
                neighbours_idx_[i,:] = n[:neighbours]
            else: 
                neighbours_idx_[i,:len(n)] = n

        neighbours_idx = neighbours_idx_

        em = np.empty([1,x_hat_train.shape[1], x_hat_train.shape[2] ,x_hat_train.shape[-1]])
        em[:] = np.nan

        x_hat_train = np.concatenate([x_hat_train, em])#if no neighbours make error large
        
    return neighbours_dist, neighbours_idx, x_hat_train, neighbour_mask

def get_nln_errors(model,
                   model_type,
                   z_query,
                   z,
                   test_images,
                   x_hat_train,
                   neighbours_idx,
                   neighbour_mask,
                   args):
    """
        Calculates the NLN error for either frNN or KNN

        Parameters
        ----------
        model (tuple): the model used
        model_type (str): the type of model (AE,VAE,...)
        z_query (np.array): test set latent space 
        z (np.array): training set latent space
        test_images (np.array): test data
        x_hat_train (np.array): reconstruction of training data
        neighbour_idx (np.array): indexes of nearest neighbours 
        neighbours_mask (np.array): boolean mask of nearest neighbours array
        args (Namespace)
        
        Returns
        -------
        np.array

    """

    x_hat  = infer(model[0], test_images, args, 'AE')
    test_images_stacked = np.stack([test_images]*neighbours_idx.shape[-1],axis=1)
    neighbours = x_hat_train[neighbours_idx]

    if ((model_type == 'AE') or 
        (model_type == 'AE_SSIM') or
        (model_type == 'AAE') or
        (model_type == 'DAE') or
        (model_type == 'VAE')):

        error_nln = get_error(model_type, test_images_stacked,neighbours,mean=False) 
        error = np.nanmean(error_nln, axis =1) #nanmean for frNN 

        error_recon = get_error(model_type, test_images, x_hat, mean=False) 

        error[neighbour_mask] = error_recon[neighbour_mask]

    elif model_type == 'DAE_disc':
        disc_x_hat_train = infer(model[1], x_hat_train, args, 'disc')
        disc_x = infer(model[1], test_images, args, 'disc')
        disc_x_hat  = infer(model[1], x_hat, args, 'disc')

        disc_x_stacked = np.stack([disc_x]*neighbours_idx.shape[-1],axis=1)
        disc_neighbours = disc_x_hat_train[neighbours_idx]

        error_nln = get_error(model_type, 
                              test_images_stacked,
                              neighbours,
                              d_x = disc_x_stacked, 
                              d_x_hat = disc_neighbours, 
                              mean=False) 
        error = np.nanmean(error_nln, axis =1) #nanmean for frNN 

        error_recon = get_error(model_type, 
                                test_images,
                                x_hat,
                                d_x = disc_x, 
                                d_x_hat = disc_x_hat, 
                                mean=False) 

        error[neighbour_mask] = error_recon[neighbour_mask]


    elif model_type == 'GANomaly':
        x_hat = infer(model[0], test_images, args, 'AE')
        z_hat = infer(model[2], x_hat, args, 'encoder')

        z_neighbours = z[neighbours_idx]
        z_hat_stacked = np.stack([z_hat]*neighbours_idx.shape[-1],axis=1)

        error_nln = get_error(model_type, 
                              test_images_stacked,
                              neighbours,
                              z = z_neighbours, 
                              z_hat = z_hat_stacked, 
                              mean=False) 

        error = np.nanmean(error_nln, axis =1) #nanmean for frNN 

        error_recon  = get_error(model_type, 
                                 test_images,
                                 x_hat,
                                 z = z_query, 
                                 z_hat = z_hat, 
                                 mean=False) 

        error[neighbour_mask] = error_recon[neighbour_mask]

    elif model_type == 'NNAE':
        x_hat = infer(model[0], test_images, args, 'AE')
        z = infer(model[0].encoder, test_images, args, 'encoder')
        z_hat = infer(model[1], [x_hat, neighbours], args, 'NNAE')

        error = np.abs(z - z_hat)

    elif model_type == 'RESNET_AE':

        #TODO: Inefficeint way of doing things
        test_images = resnet(test_images).numpy()
        error = [] 
        for n in range(neighbours.shape[1]):
            error.append(test_images - resnet(neighbours[:,n,:]).numpy())
        error = np.array(error)
        error = np.swapaxes(error, 0, 1)
            
        error = np.nanmean(error, axis =1) #nanmean for frNN 

    return error


def get_nln_metrics(model,
                    train_images,
                    test_images,
                    test_labels,
                    model_type,
                    args):
    """
        Calculates the NLN metrics for either frNN or KNN

        Parameters
        ----------
        model (tuple): the model used
        train_images (np.array): images from training set 
        test_images (np.array): testing images
        test_labels (np.array): labels of testing images
        model_type (str): the type of model (AE,VAE,...)
        args (Namespace): the argumenets from cmd_args
        
        Returns
        -------
        np.array

    """

    z_query = infer(model[0].encoder, test_images, args, 'encoder') 
    z = infer(model[0].encoder, train_images, args, 'encoder')
    x_hat_train = infer(model[0], train_images, args, 'AE')

    d,max_auc,max_f1,max_neighbours,max_radius,index_counter = {},0,0,0,0,0
    
    for n_bour in args.neighbors:
        if args.algorithm == 'knn':
            t = time()
            neighbours_dist, neighbours_idx, x_hat_train, neighbour_mask = nln(z, 
                                                                               z_query, 
                                                                               x_hat_train, 
                                                                               args.algorithm, 
                                                                               n_bour, 
                                                                               radius=None)
            error = get_nln_errors(model,
                                   model_type,
                                   z_query,
                                   z,
                                   test_images,
                                   x_hat_train,
                                   neighbours_idx,
                                   neighbour_mask,
                                   args)

            if args.patches:  
                if error.ndim ==4:
                    error, test_labels_ = reconstruct(error, args, test_labels) 
                else:
                    error, test_labels_ = reconstruct_latent_patches(error, args, test_labels) 


            error = np.nanmean(error,axis=tuple(range(1,error.ndim)))
            temp_args = [error,test_labels_,args.anomaly_class,args.neighbors,
                         [float('nan')],n_bour,float('nan'), max_auc,max_f1,max_neighbours,
                          max_radius,index_counter,d,t]

            (max_auc,max_f1,max_neighbours,
                    max_radius,index_counter,d) =  get_max_score(temp_args)

        elif args.algorithm == 'frnn':
            for r in  args.radius:
                t = time()
                neighbours_dist, neighbours_idx, x_hat_train, neighbour_mask = nln(z, 
                                                                                   z_query, 
                                                                                   x_hat_train, 
                                                                                   args.algorithm, 
                                                                                   n_bour, 
                                                                                   radius=r)
                error = get_nln_errors(model,
                                       model_type,
                                       z_query,
                                       z,
                                       test_images,
                                       x_hat_train,
                                       neighbours_idx,
                                       neighbour_mask,
                                       args)

                if args.patches:  
                    if error.ndim ==4:
                        error, test_labels_ = reconstruct(error, args, test_labels) 
                    else:
                        error, test_labels_ = reconstruct_latent_patches(error, args, test_labels) 

                error = np.mean(error,axis=tuple(range(1,error.ndim)))

                temp_args = [error,test_labels_,args.anomaly_class,
                             args.neighbors, args.radius,n_bour,r, max_auc,
                             max_f1,max_neighbours,max_radius,index_counter,d,t]

                (max_auc,max_f1,max_neighbours,
                        max_radius,index_counter,d) = get_max_score(temp_args)

    with open('outputs/{}/{}/{}/latent_scores.pkl'.format(model_type,args.anomaly_class,args.model_name),'wb') as f:
        pickle.dump(d,f)

    return max_auc,max_f1,max_neighbours,max_radius

def get_max_score(args):
    """
    TODO: make work for MVTEC
        Find the maximum AUROC score for the model 

        args (Namespace): arguments from cmd_args
    """
    (error,test_labels,anomaly,n_bours,
           radius,n_bour,rad,max_auc,max_f1,
           max_neighbours,max_radius,
           index_counter,d,t) = args
    

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
