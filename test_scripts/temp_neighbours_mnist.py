import tensorflow as tf
import numpy as np
from utils.data import *
from models import Autoencoder
from utils.metrics.latent_reconstruction import * 
from matplotlib import pyplot as plt
from data import load_cifar10, load_mnist
from sklearn import metrics

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

args = Namespace(input_shape=(28,28,1),
                  rotate=False,
                  crop=False,
                  patches=False,
                  percentage_anomaly=0,
                  limit= 1000,
                  latent_dim=500,
                  # NLN PARAMS 
                  anomaly_class= 0, 
                  data= 'MNIST',
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
            ax[i,col].imshow(test_images[indx,...,0]); 
            ax[i,col].set_title('Input image for class {}'.format(i))
            col+=1
            
            for n, dist in zip(neighbours[indx], neighbours_dist[indx]): 
                ax[i,col].imshow(n[...,0])
                ax[i,col].set_title('neigh {}, dist {}'.format(col, round(dist,2)), fontsize=6)
                col+=1
            col = args.neighbours[0]+1

            ax[i,col].imshow(recon_avg_neighbours[indx][...,0])
            ax[i,col].set_title('Average error')
            col+=1
            ax[i,col].imshow(error[indx][...,0])
            ax[i,col].set_title('Per image Average error')

        plt.tight_layout()
        plt.savefig('/tmp/neighbours/seed_{}'.format(r))
        plt.close('all')



def main():
    # Load data 
    (train_dataset, train_images, train_labels, test_images, test_labels) = load_mnist(args)

    # Load model 
    ae = Autoencoder(args)
    ae.load_weights('/tmp/temp/training_checkpoints/checkpoint_full_model_ae')

    x_hat = ae(test_images).numpy()

    x_hat_train = ae(train_images).numpy()

    z_query,error_query = get_error('AE', ae, test_images, return_z = True)
    z,_ = get_error('AE', ae, train_images, return_z = True)

    nbrs = neighbors.NearestNeighbors(n_neighbors= args.neighbours[0],
                                      algorithm='ball_tree',
                                      n_jobs=-1).fit(z) # using radius

    neighbours_dist, neighbours_idx =  nbrs.kneighbors(z_query,return_distance=True)#KNN

###
    test_images_stacked = np.stack([test_images]*neighbours_idx.shape[-1],axis=1)
    neighbours = x_hat_train[neighbours_idx]
    error = np.mean(np.abs(test_images_stacked - neighbours), axis =1)
    recon_avg_neighbours = np.mean(neighbours, axis =1)

    error_agg =  np.mean(error,axis=tuple(range(1,error.ndim)))
    fpr, tpr, thr  = metrics.roc_curve(test_labels==args.anomaly_class,error_agg)
    a_u_c = metrics.auc(fpr, tpr)
    print(a_u_c)

    KNN_encoding = ae.encoder(error)

    #plot_diff(test_images, neighbours, neighbours_idx, error_hat, error_orig_hat)

    #plot(test_images,n_patches,x_out,masks_patches,x_hat)
    
    plot_neighs(test_images, test_labels, neighbours, error,  neighbours_idx, neighbours_dist,recon_avg_neighbours)

if __name__ == '__main__':
    main()
