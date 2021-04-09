import tensorflow as tf
import numpy as np
from sklearn import metrics 
from matplotlib import pyplot as plt

import sys
sys.path.insert(1,'../')

from utils.data import *
from models import Autoencoder, Discriminator_x
from utils.metrics.latent_reconstruction import * 
from utils.data import process
from data import load_mvtec, load_cifar10, load_mnist
class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
PATCH = 256 
args = Namespace(input_shape=(PATCH, PATCH, 3),
                   rotate=False,
                   crop=False,
                   patches=True,
                   percentage_anomaly=0,
                   limit= 6,
                   patch_x = PATCH,
                   patch_y=PATCH,
                   patch_stride_x = PATCH,
                   patch_stride_y = PATCH,
                   latent_dim=1024,
                   # NLN PARAMS
                   anomaly_class='cable',
                   data= 'MVTEC',
                   neighbours = [3],
                   radius= [5],
                   algorithm = 'knn')

def plot_recon(test_images, test_labels, test_masks, x_hat, neighbours, error, error_nln): 
    test_images, test_labels = reconstruct(test_images,args,test_labels)
    test_masks = reconstruct(np.expand_dims(test_masks,axis=-1),args)[...,0]
    x_hat = reconstruct(x_hat,args)
    neighbours= reconstruct(neighbours,args)
    error_recon = reconstruct(error,args)
    error_nln_recon = reconstruct(error_nln,args)

    fig, ax = plt.subplots(args.limit, 7, figsize=(10,10))
    for i in range(args.limit):
        col = 0
        ax[i,col].imshow(test_images[i,...], vmin=0, vmax =1); 
        ax[i,col].set_title('Input image for class {}'.format(test_labels[i]),fontsize=6)
        col+=1

        ax[i,col].imshow(test_masks[i,...], vmin=0, vmax =1); 
        ax[i,col].set_title('Input Mask for class {}'.format(test_labels[i]),fontsize=6)
        col+=1

        ax[i,col].imshow(x_hat[i,...], vmin=0, vmax =1); 
        ax[i,col].set_title('Reconstruct' ,fontsize=6)
        col+=1

        ax[i,col].imshow(neighbours[i,...] , vmin=0, vmax =1); 
        ax[i,col].set_title('Average neighbours' ,fontsize=6)
        col+=1

        ax[i,col].imshow(error_recon[i,...], vmin=0, vmax =1)
        ax[i,col].set_title('Reconstructed error', fontsize=6)
        ax[i,col].set_xlabel(round(np.mean(error_recon[i,...]),4))
        col+=1

        ax[i,col].imshow(error_nln_recon[i,...], vmin=0, vmax =1)
        ax[i,col].set_title('KNN Average error' , fontsize=6)
        ax[i,col].set_xlabel(round(np.mean(error_nln_recon[i,...]),4))
        col+=1
        
        e = np.abs(x_hat[i,...] - neighbours[i,...])
        ax[i,col].imshow(e, vmin=0, vmax =1)
        ax[i,col].set_title('Recon - KNN' , fontsize=6)
        col+=1

    plt.tight_layout()
    plt.savefig('/tmp/neighbours/seed_{}'.format(i))
    plt.close('all')


def plot_neighs(test_images, test_labels, neighbours, error,  neighbours_idx, neighbours_dist,recon_avg_neighbours):
    rs = np.random.randint(0,len(test_labels),10) 
    for r in rs:
        fig, ax = plt.subplots(np.unique(test_labels).shape[0], args.neighbours[0]+3, figsize=(10,10))

        for i, j in enumerate(np.unique(test_labels)):
            col = 0

            ax[i,col].imshow(test_images[r,...]); 
            ax[i,col].set_title('Input for class {}'.format(j), fontsize=6)
            col+=1
            for n, dist in zip(neighbours[r], neighbours_dist[r]): 
                ax[i,col].imshow(n)
                ax[i,col].set_title('neigh {}, dist {}'.format(col, round(dist,2)), fontsize=6)
                col+=1
            col = args.neighbours[0]+1

            ax[i,col].imshow(recon_avg_neighbours[r])
            ax[i,col].set_title('Average Neighbours ',fontsize=5)
            col+=1

            ax[i,col].imshow(error[r])
            ax[i,col].set_title('Per image Average error',fontsize=5)
            ax[i,col].set_xlabel('Average dist {}'.format(np.mean(neighbours_dist[r])),fontsize=6)

        plt.tight_layout()
        plt.savefig('/tmp/neighbours/seed_{}'.format(r))
        plt.close('all')

def plot_dists(test_images,test_labels, test_masks, neighbours_dist):
    # restructure dists to reshape into images
    dists =  np.min(neighbours_dist,axis=-1)# Could also be max distance??
    dists = np.array([[d]*args.patch_x**2 for i,d in enumerate(dists)]).reshape(len(dists), args.patch_x, args.patch_y)#divide by mean(d) to normliase by amplitude
    dists_recon = reconstruct(np.expand_dims(dists,axis=-1),args)[...,0]
    dists_recon = process(dists_recon, per_image=False)

    test_images, test_labels = reconstruct(test_images,args,test_labels)
    test_masks = reconstruct(np.expand_dims(test_masks,axis=-1),args)[...,0]
    

    fig, ax = plt.subplots(args.limit, 3, figsize=(10,10))
    for i in range(args.limit):
        col = 0
        ax[i,col].imshow(test_images[i,...], vmin=0, vmax =1); 
        ax[i,col].set_title('Input image for class {}'.format(test_labels[i]),fontsize=6)
        col+=1

        ax[i,col].imshow(test_masks[i,...], vmin=0, vmax =1); 
        ax[i,col].set_title('Input Mask for class {}'.format(test_labels[i]),fontsize=6)
        col+=1

        ax[i,col].imshow(dists_recon[i,...])#, vmin=mi, vmax =ma); 
        ax[i,col].set_title('Distance map for class {}'.format(test_labels[i]),fontsize=6)

    plt.tight_layout()
    plt.savefig('/tmp/neighbours/dists')
    plt.close('all')

def plot_recon_neighs(test_images, test_labels, test_masks,  neighbours, neighbours_dist):
    test_images, test_labels = reconstruct(test_images,args,test_labels)
    test_masks = reconstruct(np.expand_dims(test_masks,axis=-1),args)[...,0]

    fig,axs = plt.subplots(len(test_labels), 2+args.neighbours[0],figsize=(10,10))

    for i,test_image in enumerate(test_images):
        axs[i,0].imshow(test_image) 
        axs[i,0].set_title('Input image {}'.format(i),fontsize=5)
        axs[i,1].imshow(test_masks[i]) 

        for j in range(neighbours.shape[1]):
            neighbour = reconstruct(neighbours[:,j,...],args)
            axs[i,j+2].imshow(neighbour[i,...])
            axs[i,j+2].set_title('Neighbour {}'.format(j),fontsize=5)

    plt.show()

def plot_recon_neighs_diff(test_images, test_labels, test_masks,  neighbours, neighbours_dist):
    n_patches = sizes[str(args.anomaly_class)]//args.patch_x
    test_images, test_labels = reconstruct(test_images,args,test_labels)
    test_masks = reconstruct(np.expand_dims(test_masks,axis=-1),args)[...,0]

    fig,axs = plt.subplots(len(test_labels), 2+args.neighbours[0] +1,figsize=(10,10))

    for i,test_image in enumerate(test_images):
        col = 0
        axs[i,col].imshow(test_image) 
        axs[i,col].set_title('Input image {}'.format(i),fontsize=5)
        col+=1

        axs[i,col].imshow(test_masks[i],vmin=0,vmax=1) 
        col+=1

        for j in range(args.neighbours[0]):
            dist = round(np.mean(neighbours_dist[(i)*(16**2):(i+1)*(16**2),j]),3)
            neighbour = reconstruct(neighbours[:,j,...],args)
            axs[i,j+col].imshow(neighbour[i,...],vmin=0,vmax=1)
            axs[i,j+col].set_title('Neighbour {}, mean dist ={}'.format(j,dist),fontsize=5)
        col+=j+1
#        diff = np.abs(reconstruct(neighbours[:,7,...],args) - reconstruct(neighbours[:,8,...],args))

#        axs[i,col].imshow(diff[i,...],vmin=0,vmax=0.5)
#        axs[i,col].set_title('diff 7 - 2'.format(j,dist),fontsize=5)


    plt.savefig('/tmp/neighbours/neighbours_diffs')
    plt.show()


def main():
    # Load data 
    (train_dataset, train_images, train_labels, test_images, test_labels,test_masks) = load_mvtec(args)

    # Load model 
    ae = Autoencoder(args)
    disc = Discriminator_x(args)
#    ae.load_weights('/tmp/military-orange-kiwi-of-foundation/training_checkpoints/checkpoint_full_model_ae') #128x128
#    ae.load_weights('/tmp/DAE/aquatic-cherry-bettong-of-gaiety/training_checkpoints/checkpoint_full_model_ae') #128x128 log 
#    ae.load_weights('/tmp/voracious-competent-dodo-of-memory/training_checkpoints/checkpoint_full_model_ae') #32x32
#    ae.load_weights('/tmp/DAE/successful-fair-lizard-of-certainty/training_checkpoints/checkpoint_full_model_ae') #32x32 
#    ae.load_weights('/tmp/DAE/refreshing-celadon-bulldog-from-arcadia/training_checkpoints/checkpoint_ae') #32x32 log
#    ae.load_weights('/tmp/DAE/independent-copper-lyrebird-from-tartarus/training_checkpoints/checkpoint_full_model_ae') #32x32 log ld =50
#    ae.load_weights('/tmp/DAE/garrulous-pretty-hyrax-of-completion/training_checkpoints/checkpoint_full_model_ae') #64x64
    #ae.load_weights('/tmp/DAE/eager-glorious-quail-of-ampleness/training_checkpoints/checkpoint_full_model_ae') #64x64
#    ae.load_weights('/tmp/DAE/grinning-mature-chamois-of-force/training_checkpoints/checkpoint_full_model_ae') #64x64 no crop 
#    ae.load_weights('/tmp/DAE/furry-industrious-macaque-of-fruition/training_checkpoints/checkpoint_ae') #128 log no crop
#    ae.load_weights('/tmp/DAE/witty-fearless-chachalaca-of-romance/training_checkpoints/checkpoint_full_model_ae') #64x64 non_log

    #ae.load_weights('/tmp/DAE/vigilant-fractal-gorilla-of-hail/training_checkpoints/checkpoint_full_model_ae') #128 log no crop
    #disc.load_weights('/tmp/DAE/vigilant-fractal-gorilla-of-hail/training_checkpoints/checkpoint_full_model_disc') #128 log no crop
    ae.load_weights('/tmp/AE/invisible-wandering-bandicoot-of-order/training_checkpoints/checkpoint_ae') #256x256 log ld =1024

    x_hat = ae(test_images).numpy()
    #disc_x_hat, disc_x_hat_cl = disc(x_hat)
    
    error = np.abs(x_hat - test_images)
    error_recon,recon_labels = reconstruct(error,args,test_labels)
    error_agg =  np.nanmean(error_recon,axis=tuple(range(1,error_recon.ndim)))
    fpr, tpr, thr  = metrics.roc_curve(recon_labels==args.anomaly_class,error_agg)
    a_u_c = metrics.auc(fpr, tpr)
    print("AUC for ORIG {}".format(a_u_c))

    x_hat_train = ae(train_images).numpy()

    z_query,error_query = get_error('AE', ae, test_images, return_z = True)
    z,_ = get_error('AE', ae, train_images, return_z = True)
    
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
###

    test_images_stacked = np.stack([test_images]*neighbours_idx.shape[-1],axis=1)
    neighbours = x_hat_train[neighbours_idx]
    error = np.nanmean(np.abs(test_images_stacked - neighbours), axis =1)
#### WITHOUT NORM 
    error_recon,recon_labels = reconstruct(error,args,test_labels)
    error_agg =  np.nanmean(error_recon,axis=tuple(range(1,error_recon.ndim)))
    fpr, tpr, thr  = metrics.roc_curve(recon_labels==args.anomaly_class,error_agg)
    a_u_c = metrics.auc(fpr, tpr)
    print("AUC for NLN without NORM {}".format(a_u_c))
#### ERROR WITH NEIGHBOURS 
    #error_n = np.abs(neighbours[:,7,...]- neighbours[:,8,...])

    #error_recon_n,recon_labels_n = reconstruct(error_n,args,test_labels)
    #error_agg_ =  np.nanmean(error_recon_n,axis=tuple(range(1,error_recon.ndim)))
    #fpr, tpr, thr  = metrics.roc_curve(recon_labels_n==args.anomaly_class,error_agg_)
    #a_u_c = metrics.auc(fpr, tpr)
    #print("AUC for Niehgbours 0 and 1 {}".format(a_u_c))
#### WITH NORM 
    #error = process(error,per_image=False)
    dists = process(neighbours_dist, per_image=False)

    for i, d in enumerate(test_images):#normliase error by amplitude 
        error[i] = error[i]/np.mean(d)
        
    neighbours_mean= np.nanmean(neighbours, axis =1)

    error_recon,recon_labels = reconstruct(error,args,test_labels)
    error_agg =  np.nanmean(error_recon,axis=tuple(range(1,error_recon.ndim)))
    fpr, tpr, thr  = metrics.roc_curve(recon_labels==args.anomaly_class,error_agg)
    a_u_c = metrics.auc(fpr, tpr)
    print("AUC for NLN with NORM {}".format(a_u_c))



    #plot_diff(test_images, neighbours, neighbours_idx, error_hat, error_orig_hat)

    #plot(test_images,n_patches,x_out,masks_patches,x_hat)
    
    plot_neighs(test_images, test_labels, neighbours, error,  neighbours_idx, neighbours_dist,neighbours_mean)
    plot_recon(test_images, test_labels, test_masks, x_hat, neighbours_mean, np.abs(test_images -x_hat), error) 
    plot_dists(test_images,test_labels, test_masks, neighbours_dist)
    plot_recon_neighs_diff(test_images, test_labels, test_masks,  neighbours, neighbours_dist)

if __name__ == '__main__':
    main()
