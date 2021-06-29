import tensorflow as tf
import numpy as np
from utils.data import *
from models import Autoencoder, Discriminator_x
from utils.metrics.latent_reconstruction import * 
from utils.data import process, sizes
from matplotlib import pyplot as plt
from data import load_mvtec, load_cifar10, load_mnist
from sklearn import metrics 

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

PATCH = 128 
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
                   latent_dim=6000,
                   # NLN PARAMS
                   anomaly_class='cable',
                   data= 'MVTEC',
                   neighbours = [5],
                   radius= [5],
                   algorithm = 'knn')

def plot_discs(test_images, test_masks, disc_x,  x_hat, disc_x_hat, discs_neighbours): 
    test_images = reconstruct(test_images,args)
    test_masks = reconstruct(np.expand_dims(test_masks,axis=-1),args)[...,0]
    x_hat = reconstruct(x_hat,args)

    n_patches = sizes[str(args.anomaly_class)]//args.patch_x
    strt, fnnsh = 0, n_patches**2 

    disc_x = process(disc_x,False)
    disc_x_hat = process(disc_x_hat, False)
    disc_neighbours= process(discs_neighbours,False)

    fig, ax = plt.subplots(args.limit, 5 + args.neighbours[0], figsize=(10,10))
    for i in range(args.limit):
        col = 0
        ax[i,col].imshow(test_images[i,...], vmin=0, vmax =1); 
        ax[i,col].set_title('Input image',fontsize=6)
        col+=1

        ax[i,col].imshow(test_masks[i,...], vmin=0, vmax =1); 
        ax[i,col].set_title('Input Mask',fontsize=6)
        col+=1

        ax[i,col].imshow(disc_x[strt:fnnsh].reshape([n_patches, n_patches]), vmin=0, vmax =1); 
        ax[i,col].set_title('Input Disc', fontsize=6)
        col+=1

        ax[i,col].imshow(x_hat[i,...], vmin=0, vmax =1); 
        ax[i,col].set_title('Reconstruct' ,fontsize=6)
        col+=1

        ax[i,col].imshow(disc_x_hat[strt:fnnsh,:].reshape([n_patches ,n_patches]), vmin=0, vmax =1); 
        ax[i,col].set_title('Discriminator on x_hat' ,fontsize=6)
        col+=1

        for n in range(args.neighbours[0]):
            img = discs_neighbours[strt:fnnsh, n,:].reshape([n_patches,n_patches])
            ax[i,col+n].imshow(img,vmin=0, vmax =1)

        strt=fnnsh
        fnnsh = fnnsh+n_patches**2
    
    plt.show()
    plt.savefig('/tmp/neighbours/seed_{}'.format(i))
    plt.close('all')

def plot_discs_z(test_images, test_masks, disc_x, x_hat, discs_x_hat, discs_neighbours): 
    test_images = reconstruct(test_images,args)
    test_masks = reconstruct(np.expand_dims(test_masks,axis=-1),args)[...,0]
    x_hat = reconstruct(x_hat,args)

    strt, fnnsh =0, 64
    mi,ma = -5, 5

    fig, ax = plt.subplots(args.limit, 5+args.neighbours[0], figsize=(10,10))
    for i in range(args.limit):
        col = 0
        ax[i,col].imshow(test_images[i,...], vmin=0, vmax =1); 
        ax[i,col].set_title('Input image',fontsize=6)
        col+=1

        ax[i,col].imshow(test_masks[i,...], vmin=0, vmax =1); 
        ax[i,col].set_title('Input Mask',fontsize=6)
        col+=1

        ax[i,col].plot(disc_x[strt:fnnsh,:].flatten())#, vmin=0, vmax =1); 
        ax[i,col].set_ylim([mi, ma])
        ax[i,col].set_title('Disc Input',fontsize=6)
        col+=1

        ax[i,col].imshow(x_hat[i,...], vmin=0, vmax =1); 
        ax[i,col].set_title('Reconstruct' ,fontsize=6)
        col+=1

#        discs_ = discs_x_hat[strt:fnnsh,:].reshape(640,600)
        ax[i,col].plot(discs_x_hat[strt:fnnsh,:].flatten())# , vmin=0, vmax =1); 
        ax[i,col].set_title('Discriminator on x_hat' ,fontsize=6)
        ax[i,col].set_ylim([mi, ma])
        col+=1

        for n in range(args.neighbours[0]):
            # do some magic numbers to reshape
            # every 64 correnspond to a new image, note this will break for patches that arent 128x128 and 6000
#            discs = discs_neighbours[strt:fnnsh, n, :].reshape(640,600)
            ax[i,col+n].plot(discs_neighbours[strt:fnnsh, n, :].flatten())#, vmin=0, vmax =1)
            ax[i,col+n].set_ylim([mi, ma])

        strt = fnnsh
        fnnsh +=64

    #plt.tight_layout()
    plt.savefig('/tmp/neighbours/seed_{}'.format(i))
    plt.show()
    plt.close('all')

def plot_dists(test_images, test_masks, x_hat, neighbours, neighbours_dist): 
    test_images = reconstruct(test_images,args)
    test_masks = reconstruct(np.expand_dims(test_masks,axis=-1),args)[...,0]
    x_hat = reconstruct(x_hat,args)

    fig, ax = plt.subplots(args.limit, 3 + args.neighbours[0], figsize=(10,10))
    for i in range(args.limit):
        col = 0
        ax[i,col].imshow(test_images[i,...], vmin=0, vmax =1); 
        ax[i,col].set_title('Input image',fontsize=6)
        col+=1

        ax[i,col].imshow(test_masks[i,...], vmin=0, vmax =1); 
        ax[i,col].set_title('Input Mask',fontsize=6)
        col+=1

        ax[i,col].imshow(x_hat[i,...], vmin=0, vmax =1); 
        ax[i,col].set_title('Reconstruct' ,fontsize=6)
        col+=1

        for j in range(args.neighbours[0]):
            neighbour = reconstruct(neighbours[:,j,...] * neighbours_dist[:,0].reshape(neighbours.shape[0], 1, 1, 1),
                                    args)
            neighbour = process(neighbour, True)
            ax[i, col + j].imshow(neighbour[i,...], vmin =0, vmax= 1)
            ax[i, col + j].set_title('Neighbour {}'.format(j),fontsize=5)

    plt.savefig('/tmp/neighbours/seed_{}'.format(i))
    plt.show()
    plt.close('all')

def nln(z, z_query, x_hat_train):
    if args.algorithm == 'knn':
        nbrs = neighbors.NearestNeighbors(n_neighbors= args.neighbours[0],
                                          algorithm='ball_tree',
                                          n_jobs=-1).fit(z) # using radius

        neighbours_dist, neighbours_idx =  nbrs.kneighbors(z_query,return_distance=True)#KNN


    elif args.algorithm == 'frnn':
        nbrs = neighbors.NearestNeighbors(radius=args.radius[0], 
                                          algorithm='ball_tree',
                                          n_jobs=-1).fit(z) # using radius

        neighbours_dist, neighbours_idx =  nbrs.radius_neighbors(z_query,
                                                  return_distance=True,
                                                  sort_results=True)#radius
        neighbours_idx_ = -1*np.ones([len(neighbours_idx), args.neighbours[0]],dtype=int)

        for i,n in enumerate(neighbours_idx):
            if len(n) == 0:
                pass 
            elif len(n) > args.neighbours[0]:
                neighbours_idx_[i,:] = n[:args.neighbours[0]]
            else: 
                neighbours_idx_[i,:len(n)] = n
        neighbours_idx = neighbours_idx_
        x_hat_train = np.concatenate([x_hat_train, np.empty([1,args.patch_x,args.patch_y,3])])#if no neighbours make error large

    return neighbours_dist, neighbours_idx, x_hat_train

def disc_neighbours(disc, neighbours_idx, x_hat_train):
    outputs_z = np.empty([len(x_hat_train), args.neighbours[0], args.latent_dim])
    outputs_cl = np.empty([len(x_hat_train), args.neighbours[0], 1])
    for i in range(len(x_hat_train)):
        z,cl = disc(x_hat_train[neighbours_idx[i]])
        outputs_z[i] = z.numpy()
        outputs_cl[i] = cl.numpy()
    return outputs_cl 

def main():
    # Load data 
    (train_dataset, train_images, train_labels, test_images, test_labels,test_masks) = load_mvtec(args)

    # Load model 
    ae = Autoencoder(args)
    disc = Discriminator_x(args)

    ae.load_weights('/tmp/DAE/vigilant-fractal-gorilla-of-hail/training_checkpoints/checkpoint_full_model_ae') #128 log no crop
    disc.load_weights('/tmp/DAE/vigilant-fractal-gorilla-of-hail/training_checkpoints/checkpoint_full_model_disc') #128 log no crop

    x_hat = ae(test_images).numpy()
    disc_x_hat, disc_x_hat_cl = disc(x_hat)
    disc_x, disc_x_cl = disc(test_images)

    ##################################################
    ##################################################
    ####### WORK ON THE SUBTRACTION ##################
    ##################################################
    ##################################################
    #disc_x_stacked = np.stack([disc_x.numpy()]*neighbours_idx.shape[-1],axis=1)
    ##################################################
    ##################################################
    ##################################################
    ##################################################
    ##################################################
    
    x_hat_discs = disc_x_hat_cl.numpy()

    x_hat_train = ae(train_images).numpy()
    z_query,error_query = get_error('AE', ae, test_images, return_z = True)
    z,_ = get_error('AE', ae, train_images, return_z = True)
    
    neighbours_dist, neighbours_idx, x_hat_train = nln(z, z_query, x_hat_train)

    test_images_stacked = np.stack([test_images]*neighbours_idx.shape[-1],axis=1)
    neighbours = x_hat_train[neighbours_idx]
    error = np.nanmean(np.abs(test_images_stacked - neighbours), axis =1)
#### WITHOUT NORM 
    error_recon,recon_labels = reconstruct(error,args,test_labels)
    error_agg =  np.nanmean(error_recon,axis=tuple(range(1,error_recon.ndim)))
    fpr, tpr, thr  = metrics.roc_curve(recon_labels==args.anomaly_class,error_agg)
    a_u_c = metrics.auc(fpr, tpr)
    print("AUC for NLN without NORM {}".format(a_u_c))

    discs_neighbours = disc_neighbours(disc, neighbours_idx, x_hat_train)
    
    plot_discs(test_images, test_masks, disc_x_cl.numpy(), x_hat, x_hat_discs, discs_neighbours) 

if __name__ == '__main__':
    main()
