import tensorflow as tf
import numpy as np
from utils.data import *
from models import Autoencoder
from utils.metrics.latent_reconstruction import * 
from sklearn.metrics import roc_curve, auc, f1_score
from matplotlib import pyplot as plt
from data import load_cifar10

LIMIT = 1000

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

args_patches  = Namespace(input_shape=(8,8,3),
                  rotate=False,
                  crop=True,
                  patches=True,
                  patch_x = 8,
                  patch_y=8,
                  patch_stride_x = 8,
                  patch_stride_y = 8,
                  percentage_anomaly=0,
                  limit= None,
                  latent_dim=500,
                  # NLN PARAMS 
                  anomaly_class= 1, 
                  data= 'CIFAR10',
                  neighbours = [5],
                  radius = [4.0],
                  algorithm = 'knn')


args_non = Namespace(input_shape=(32,32,3),
                  rotate=False,
                  crop=False,
                  patches=False,
                  percentage_anomaly=0,
                  limit= None,
                  latent_dim=500,
                  # NLN PARAMS 
                  anomaly_class= 1, 
                  data= 'CIFAR10',
                  neighbours = [5],
                  radius = [4.0],
                  algorithm = 'knn')

def plot_diff(test_labels, test_images, x_hat, nln_x_hat, nln_x_hat_patches, nln_error, nln_error_patches): 
    lim = nln_error_patches.shape[0]

    _nln =  nln_error.mean(axis=tuple(range(1,nln_error.ndim)))
    fpr,tpr,thr =  roc_curve(test_labels[:lim]==1, _nln[:lim])
    _nln_auc = auc(fpr,tpr)

    _nln_patches =  nln_error_patches.mean(axis=tuple(range(1,nln_error_patches.ndim)))
    fpr,tpr,thr =  roc_curve(test_labels[:lim]==1, _nln_patches[:lim])
    _nln_patches_auc = auc(fpr,tpr)


    for _ in range(4):
        fig, ax = plt.subplots(10, 6, figsize=(10,10))
        r = np.random.randint(0,30)
        for i in range(10):
            col = 0
            indx = np.where(test_labels==i)[0][r]

            ax[i,col].imshow(test_images[indx,...], vmin=0, vmax =1); 
            ax[i,col].set_title('Input image for class {}'.format(i),fontsize=6)
            col+=1

            ax[i,col].imshow(x_hat[indx,...], vmin=0, vmax =1); 
            ax[i,col].set_title('Reconstruction',fontsize=6)
            col+=1

            ax[i,col].imshow(nln_x_hat[indx,...] , vmin=0, vmax =1); 
            ax[i,col].set_title('Average NLN',fontsize=6)
            col+=1

            ax[i,col].imshow(nln_x_hat_patches[indx,...], vmin=0, vmax =1)
            ax[i,col].set_title('Average NLN patches', fontsize=6)
            col+=1

            ax[i,col].imshow(nln_error[indx,...], vmin=0, vmax =1)
            ax[i,col].set_title('Average NLN error \nAUC={}'.format(_nln_auc), fontsize=6)
            col+=1

            ax[i,col].imshow(nln_error_patches[indx,...], vmin=0, vmax =1)
            ax[i,col].set_title('Average NLN patches error \nAUC={}'.format(_nln_patches_auc),fontsize=6)
            col+=1

        plt.tight_layout()
        plt.savefig('/tmp/neighbours/seed_{}'.format(r))
        plt.close('all')



def reconstruct(x_out,test_images,args):
    t = x_out.transpose(0,2,1,3)
    n_patches = test_images.shape[1]//args.patch_x
    recon = np.empty([x_out.shape[0]//n_patches**2, args.patch_x*n_patches,args.patch_y*n_patches,x_out.shape[-1]])

    start, counter, indx, b  = 0, 0, 0, []

    for i in range(n_patches, x_out.shape[0], n_patches):
        b.append(np.reshape(np.stack(t[start:i,...],axis=0),(n_patches*args.patch_x,args.patch_x,x_out.shape[-1])))
        start = i
        counter +=1
        if counter == n_patches:
            recon[indx,...] = np.hstack(b)
            indx+=1
            counter, b = 0, []

    return recon.transpose(0,2,1,3)


def main():
    # Load data 
    (_, train_images_patches, _, test_images_patches, _) = load_cifar10(args_patches)
    (_, train_labels), (_, test_labels) = tf.keras.datasets.cifar10.load_data()# to not have to rejoin patches
    test_images_patches, test_labels  = test_images_patches[:LIMIT,...], test_labels[:LIMIT,0] 
    train_images_patches, train_labels = train_images_patches[:LIMIT,...], train_labels[:LIMIT,0]

    (_, train_images, _, test_images, _) = load_cifar10(args_non)

    test_images = test_images[:LIMIT,...]
    train_images = train_images[:LIMIT,...]

    # Load model 
    ae_patches = Autoencoder(args_patches)
    ae_patches.load_weights('/tmp/temp/training_checkpoints/checkpoint_full_model_ae')
    x_hat_patches = ae_patches(test_images_patches).numpy()
    x_hat_train_patches = ae_patches(train_images_patches).numpy()
    z_query_patches,error_query_patches = get_error('AE', ae_patches, test_images_patches, return_z = True)
    z_patches,_ = get_error('AE', ae_patches, train_images_patches, return_z = True)
    nbrs_patches = neighbors.NearestNeighbors(n_neighbors=args_patches.neighbours[0],
                                     algorithm='ball_tree',
                                      n_jobs=-1).fit(z_patches) # using radius
     
    neighbours_dist_patches, neighbours_idx_patches =  nbrs_patches.kneighbors(z_query_patches,return_distance=True)

    error_patches,neighbours_patches = [], []
    for i,n in enumerate(neighbours_idx_patches):
        d = x_hat_train_patches[n.astype(int)]
        neighbours_patches.append(d)
        im = np.stack([test_images_patches[i]]*n.shape[-1],axis=0)
        error_patches.append(np.mean(np.abs(d - im), axis=0))
    error_patches =np.array(error_patches)
    nln_recon_error_patches = reconstruct(error_patches,test_images,args_patches)
    nln_x_hat_patches = reconstruct(np.mean(np.array(neighbours_patches), axis = 1), test_images, args_patches)

    ae_non = Autoencoder(args_non)
    ae_non.load_weights('/tmp/temp_non/training_checkpoints/checkpoint_full_model_ae')
    x_hat= ae_non(test_images).numpy()
    x_hat_train= ae_non(train_images).numpy()
    z_query, error_query = get_error('AE', ae_non , test_images, return_z = True)
    z,_ = get_error('AE', ae_non, train_images, return_z = True)

    nbrs = neighbors.NearestNeighbors(n_neighbors=args_non.neighbours[0],
                                     algorithm='ball_tree',
                                      n_jobs=-1).fit(z) # using radius
     
    neighbours_dist, neighbours_idx =  nbrs.kneighbors(z_query,return_distance=True)

    error, neighbours = [], []
    for i,n in enumerate(neighbours_idx):
        d = x_hat_train[n.astype(int)]
        neighbours.append(d)
        im = np.stack([test_images[i]]*n.shape[-1],axis=0)
        error.append(np.mean(np.abs(d - im), axis=0))

    nln_error =np.array(error)
    nln_x_hat = np.mean(np.array(neighbours), axis = 1)

    plot_diff(test_labels, test_images, x_hat, nln_x_hat, nln_x_hat_patches, nln_error, nln_recon_error_patches)

if __name__ == '__main__':
    main()
