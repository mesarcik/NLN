import tensorflow as tf
import numpy as np
from utils.data import *
from models import Autoencoder
from utils.metrics.latent_reconstruction import * 
from matplotlib import pyplot as plt
from data import load_mvtec, load_cifar10

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
args = Namespace(input_shape=(32,32,3),
                   rotate=False,
                   crop=False,
                   patches=True,
                   percentage_anomaly=0,
                   limit= 5,
                   patch_x = 32,
                   patch_y=32,
                   patch_stride_x = 32,
                   patch_stride_y = 32,
                   latent_dim=500,
                   # NLN PARAMS
                   anomaly_class='cable',
                   data= 'MVTEC',
                   neighbours = [5],
                   algorithm = 'knn')


def plot_diff(test_images, test_labels, x_hat, error, error_hat, error_orig_hat): 
    rs = np.random.randint(0,50,10) 
    fig, ax = plt.subplots(np.unique(test_labels).shape[0], 6, figsize=(10,10))
    for i,r in enumerate(rs):
        col = 0
        indx = np.where(test_labels==i)[0][r]
        ax[i,col].imshow(test_images[indx,...], vmin=0, vmax =1); 
        ax[i,col].set_title('Input image for class {}'.format(i),fontsize=6)
        col+=1

        ax[i,col].imshow(x_hat[indx,...], vmin=0, vmax =1); 
        ax[i,col].set_title('Output image for class {}'.format(i),fontsize=6)
        col+=1

        ax[i,col].imshow(np.abs(test_images[indx,...] - x_hat[indx,...]) , vmin=0, vmax =1); 
        ax[i,col].set_title('Error image for class {}'.format(i),fontsize=6)
        col+=1

        ax[i,col].imshow(error_orig_hat[indx,...], vmin=0, vmax =1)
        ax[i,col].set_title('Reconstructed error', fontsize=6)
        col+=1

        ax[i,col].imshow(error[indx,...], vmin=0, vmax =1)
        ax[i,col].set_title('KNN Average error', fontsize=6)
        col+=1

        ax[i,col].imshow(error_hat[indx,...], vmin=0, vmax =1)
        ax[i,col].set_title('Reconstructed KNN error',fontsize=6)
        col+=1

    plt.tight_layout()
    plt.savefig('/tmp/neighbours/seed_{}'.format(r))
    plt.close('all')


def plot_neighs(test_images, test_labels, neighbours, error,  neighbours_idx, neighbours_dist,recon_avg_neighbours):
    rs = np.random.randint(0,15,10) 
    for r in rs:
        fig, ax = plt.subplots(np.unique(test_labels).shape[0], args.neighbours[0]+3, figsize=(10,10))

        for i in np.unique(test_labels):
            col = 0

            indx = np.where(test_labels==i)[0][r]
            ax[i,col].imshow(test_images[indx,...]); 
            ax[i,col].set_title('Input image for class {}'.format(i))
            col+=1
            
            for n, dist in zip(neighbours[indx], neighbours_dist[indx]): 
                ax[i,col].imshow(n)
                ax[i,col].set_title('neigh {}, dist {}'.format(col, round(dist,2)), fontsize=6)
                col+=1
            col = args.neighbours[0]+1

            ax[i,col].imshow(recon_avg_neighbours[indx])
            ax[i,col].set_title('Average error')
            col+=1
            ax[i,col].imshow(error[indx])
            ax[i,col].set_title('Per image Average error')

        plt.tight_layout()
        plt.savefig('/tmp/neighbours/seed_{}'.format(r))
        plt.close('all')



def main():
    # Load data 
    (train_dataset, train_images, train_labels, test_images, test_labels) = load_cifar10(args)

    # Load model 
    ae = Autoencoder(args)
    ae.load_weights('/tmp/temp/training_checkpoints/checkpoint_full_model_ae')


    x_hat = ae(test_images).numpy()

    x_hat_train = ae(train_images).numpy()

    z_query,error_query = get_error('AE', ae, test_images, return_z = True)
    z,_ = get_error('AE', ae, train_images, return_z = True)

    nbrs = neighbors.NearestNeighbors(radius=args.radius[0], 
                                      algorithm='ball_tree',
                                      n_jobs=-1).fit(z) # using radius

    _,neighbours_idx =  nbrs.radius_neighbors(z_query,
                                              return_distance=True,
                                              sort_results=True)#radius
    error = []
    neighbours = [] 
    z_ = np.zeros(x_hat_train.shape)
    for i,n in enumerate(neighbours_idx):
        if len(n) ==0: 
            temp = np.array([255])
            d = np.zeros(x_hat_train[0:1,...].shape)

        elif len(n) > args.neighbours[0]: 
            temp  = n[:args.neighbours[0]] 
            #d = z_[temp.astype(int)]
            d = x_hat_train[temp.astype(int)]
        else: 
            temp  = n
            #d = z_[temp.astype(int)]
            d = x_hat_train[temp.astype(int)]

        neighbours.append(d)
        im = np.stack([test_images[i]]*temp.shape[-1],axis=0)

        error.append(np.mean(np.abs(d - im), axis=0))

    error =np.array(error)

    KNN_encoding = ae.encoder(error)

    plot_diff(test_images, neighbours, neighbours_idx, error_hat, error_orig_hat)

    #plot(test_images,n_patches,x_out,masks_patches,x_hat)
    
    plot_neighs(test_images, test_labels, neighbours, neighbours_idx, neighbours_dist);

if __name__ == '__main__':
    main()
