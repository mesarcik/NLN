import tensorflow as tf
import numpy as np
from sklearn import metrics 
from matplotlib import pyplot as plt

import sys
sys.path.insert(1,'../')

from utils.data import *
from models import Autoencoder, Discriminator_x
from utils.metrics.latent_reconstruction import * 
from utils.data import process, sizes
from data import load_mvtec, load_cifar10, load_mnist

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

PATCH = 64 
LD = 1024 #500
args = Namespace(input_shape=(PATCH, PATCH, 3),
                   rotate=False,
                   crop=False,
                   patches=True,
                   percentage_anomaly=0,
                   limit= 10,
                   patch_x = PATCH,
                   patch_y=PATCH,
                   patch_stride_x = PATCH,
                   patch_stride_y = PATCH,
                   latent_dim=LD,
                   # NLN PARAMS
                   anomaly_class='cable',
                   data= 'MVTEC',
                   neighbours = [5],
                   radius= [8],
                   algorithm = 'knn')

def plot_recon_neighs(test_images, test_labels, test_masks,  x_hat, neighbours, neighbours_dist, neighbour_mask):
    test_images, test_labels = reconstruct(test_images,args,test_labels)
    test_masks = reconstruct(np.expand_dims(test_masks,axis=-1),args)[...,0]
    x_hat = reconstruct(x_hat,args)
    print(x_hat.shape)

    n_patches = sizes[args.anomaly_class]//PATCH
    strt, fnnsh = 0, n_patches**2

    fig,axs = plt.subplots(len(test_labels), 3+args.neighbours[0],figsize=(10,10))

    for i,test_image in enumerate(test_images):
        col = 0
        axs[i,col].imshow(test_image) 
        axs[i,col].set_title('Input image {}'.format(i),fontsize=5)

        col+=1
        axs[i,col].imshow(test_masks[i]) 
        axs[i,col].set_title('Input Mask',fontsize=5)

        col+=1
        axs[i,col].imshow(x_hat[i,...]) 
        axs[i,col].set_title('Reconstruction',fontsize=5)

        col+=1
        for j in range(args.neighbours[0]):
            neighbour = reconstruct(neighbours[:,j,...],args)
            cnt = np.count_nonzero(neighbour_mask[strt:fnnsh,j])

            axs[i,j+col].imshow(neighbour[i,...])
            axs[i,j+col].set_title('Neigh {}, cnt {}'.format(j,cnt),fontsize=5)

        strt = fnnsh
        fnnsh += n_patches**2

    plt.savefig('/tmp/neighbours/neighbours_recon')
    plt.show()

def nln(z, z_query, x_hat_train, mask=False):
    if args.algorithm == 'knn':
        nbrs = neighbors.NearestNeighbors(n_neighbors= args.neighbours[0],
                                          algorithm='ball_tree',
                                          n_jobs=-1).fit(z) # using radius

        neighbours_dist, neighbours_idx =  nbrs.kneighbors(z_query,return_distance=True)#KNN

        return neighbours_dist, neighbours_idx, x_hat_train

    elif args.algorithm == 'frnn':
        nbrs = neighbors.NearestNeighbors(radius=args.radius[0], 
                                          algorithm='ball_tree',
                                          n_jobs=-1).fit(z) # using radius

        neighbours_dist, neighbours_idx =  nbrs.radius_neighbors(z_query,
                                                  return_distance=True,
                                                  sort_results=True)#radius
        neighbours_idx_ = -1*np.ones([len(neighbours_idx), args.neighbours[0]],dtype=int)
        neighbour_mask  = np.zeros([len(neighbours_idx), args.neighbours[0]], dtype=bool)

        for i,n in enumerate(neighbours_idx):
            if len(n) == 0:
                neighbour_mask[i,:] = [True]*args.neighbours[0]
                pass
            elif len(n) > args.neighbours[0]:
                neighbours_idx_[i,:] = n[:args.neighbours[0]]
            else: 
                neighbours_idx_[i,:len(n)] = n
                neighbour_mask[i,len(n):] = [True]*(args.neighbours[0] - len(n))

        neighbours_idx = neighbours_idx_

        em = np.empty([1,args.patch_x,args.patch_y,3])
        em[:] = np.nan

        x_hat_train = np.concatenate([x_hat_train,em ])#if no neighbours make error large
        
        if mask:
            return neighbours_dist, neighbours_idx, x_hat_train, neighbour_mask
        else:
            return neighbours_dist, neighbours_idx, x_hat_train

def main():
    # Load data 
    (train_dataset, train_images, train_labels, test_images, test_labels,test_masks) = load_mvtec(args)

    # Load model 
    ae = Autoencoder(args)

    #ae.load_weights('/tmp/DAE/vigilant-fractal-gorilla-of-hail/training_checkpoints/checkpoint_full_model_ae') #128 log no crop
    #ae.load_weights('/tmp/DAE/grinning-mature-chamois-of-force/training_checkpoints/checkpoint_full_model_ae') #64
    ae.load_weights('/tmp/DAE/poised-savvy-porcupine-from-valhalla/training_checkpoints/checkpoint_ae') #64 more training 

    x_hat = ae(test_images).numpy()
    

    x_hat_train = ae(train_images).numpy()
    z_query,error_query = get_error('AE', ae, test_images, return_z = True)
    z,_ = get_error('AE', ae, train_images, return_z = True)
    
    neighbours_dist, neighbours_idx, x_hat_train, neighbour_mask = nln(z, z_query, x_hat_train, mask=True)

    test_images_stacked = np.stack([test_images]*neighbours_idx.shape[-1],axis=1)
    neighbours = x_hat_train[neighbours_idx]
    error = np.nanmean(np.abs(test_images_stacked - neighbours), axis =1)

    #### WITHOUT NORM 
    error_recon,recon_labels = reconstruct(error,args,test_labels)
    error_agg =  np.nanmean(error_recon,axis=tuple(range(1,error_recon.ndim)))
    fpr, tpr, thr  = metrics.roc_curve(recon_labels==args.anomaly_class,error_agg)
    a_u_c = metrics.auc(fpr, tpr)
    print("AUC for NLN without NORM {}".format(a_u_c))

    plot_recon_neighs(test_images, test_labels, test_masks,  x_hat, neighbours, neighbours_dist, neighbour_mask)
    

if __name__ == '__main__':
    main()
