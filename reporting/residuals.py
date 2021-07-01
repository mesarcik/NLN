import tensorflow as tf
import numpy as np
import os 
import pandas as pd
import pickle
from matplotlib import pyplot as plt
import sys
sys.argv = [''] 

from inference import infer, get_error
from utils.data import reconstruct,process
from utils.metrics import *
from models import Autoencoder  
from models_mvtec import Autoencoder as Autoencoder_MVTEC 

from data import *
from inference import *

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    def set_class(self,clss):
        self.anomaly_class = clss
    def set_dataset(self,dataset):
        self.data=dataset 
    def set_name(self,name):
        self.model_name=name 
    def set_input_shape(self,input_shape):
        self.input_shape= input_shape 
    def set_aug(self,b):
        self.rotate=b
        self.crop=b
        self.patches=b
    def set_latent_dim(self,ld):
        self.latent_dim = ld
    def set_mvtec_path(self,path):
        self.mvtec_path = path


PATCH = 128 

args = Namespace(
    data='MVTEC',
    seed='12334',
    input_shape=(PATCH, PATCH, 1),
    anomaly_type='MISO',
    rotate=True,
    crop=True,
    patches=True,
    percentage_anomaly=0,
    model_name=None,
    limit=None,
    patch_x = PATCH,
    patch_y=PATCH,
    patch_stride_x = PATCH,
    patch_stride_y = PATCH,
    crop_x=PATCH,
    crop_y=PATCH,
    latent_dim=128,
    # NLN PARAMS
    anomaly_class='????',
    radius= [10],
    neighbors= [2],
    algorithm = 'knn'
)

def main(cmd_args):
    df = pd.read_csv('outputs/results_{}_{}.csv'.format(cmd_args.data, cmd_args.seed)) 
    dataset = 'MVTEC'
    df = df[df.Model == 'AE']
    for cls in list(pd.unique(df.Class)):
        model_name = df[df.Class == cls].Name.item()
        args.set_dataset(dataset)
        args.set_class(cls)

        if (('grid' in cls) or
            ('screw' in cls) or 
            ('zipper' in cls)): 
            args.set_input_shape((PATCH,PATCH,1))
        else:
            args.set_input_shape((PATCH,PATCH,3))

        args.set_aug(True)
        args.set_latent_dim(128)
        args.set_mvtec_path(cmd_args.mvtec_path)
        (train_dataset, train_images, train_labels, test_images, test_labels,test_masks) = load_mvtec(args)
        ae = Autoencoder_MVTEC(args)        

        ae.load_weights('outputs/AE/{}/{}/training_checkpoints/checkpoint_full_model_ae'.format(cls,model_name))

        x_hat  = infer(ae, test_images, args, 'AE')
        x_hat_train  = infer(ae, train_images, args, 'AE')
        z_query = infer(ae.encoder, test_images, args, 'encoder') 
        z = infer(ae.encoder, train_images, args, 'encoder')

        neighbours_dist, neighbours_idx, x_hat_train, neighbour_mask = nln(z, 
                                                                           z_query, 
                                                                           x_hat_train, 
                                                                           args.algorithm, 
                                                                           args.neighbors[0], 
                                                                           radius=None)

        nln_error = get_nln_errors([ae],'AE', z_query, z, test_images, x_hat_train, neighbours_idx, neighbour_mask, args)
        error = get_error('AE', test_images, x_hat, mean=False) 
        neighbours = x_hat_train[neighbours_idx]

        x_hat_recon = reconstruct(x_hat, args)
        neighbours = x_hat_train[neighbours_idx]
        neighbours_recon = []

        for i in range(args.neighbors[0]):
            neighbours_recon.append(reconstruct(neighbours[:,i,...], args))
        neighbours_recon = np.array(neighbours_recon)

        test_images_recon, test_labels_recon = reconstruct(test_images, args,test_labels)
        nln_error_recon = reconstruct(nln_error, args)
        error_recon = reconstruct(error, args)
        masks_recon= reconstruct(np.expand_dims(test_masks,axis=-1), args)[...,0]
        dists_recon = get_dists(neighbours_dist, args)

        nln_norm = np.mean(process(nln_error_recon, per_image=False),axis=-1)
        recon_norm = np.mean(process(error_recon, per_image=False), axis=-1)
        dists_norm = process(dists_recon, per_image=False)

        ind_mvtec = 0
        if (('grid' in cls) or
            ('screw' in cls) or 
            ('zipper' in cls)): 
            test_images_recon = test_images_recon[...,0]
            x_hat_recon= x_hat_recon[...,0]
            neighbours_recon= neighbours_recon[:][...,0]
            #recon_norm = recon_norm[...,0]
            #nln_norm = nln_norm[...,0]
        fig, axs  = plt.subplots(1,4 + args.neighbors[0],figsize=(9,3))
        for offset, labels_recon in enumerate(test_labels_recon):
            if labels_recon == args.anomaly_class: ind_mvtec=offset 
            else: continue
            axs[0].imshow(test_images_recon[ind_mvtec,...],cmap='gray'); axs[0].set_title('Input',fontsize=7); axs[0].axis('off')
            axs[1].imshow(x_hat_recon[ind_mvtec,...],cmap='gray'); axs[1].set_title('Reconstruction',fontsize=7); axs[1].axis('off')

            for i in range(args.neighbors[0]):
                if i == 0: s = 'st'
                elif i == 1: s = 'nd'
                else: s = 'rd'
                axs[i+2].imshow(neighbours_recon[i][ind_mvtec,...],cmap='gray'); axs[i+2].set_title('{0}$^{{{1}}}$ Nearest-Latent-Neighbour'.format(i+1,s),fontsize=6); 
                axs[i+2].axis('off')
            
            #axs[5].imshow(dists_norm[r,...,0],vmin=0, vmax=1)#,cmap='gray'); 
            #axs[5].set_title('Distance Recon', fontsize=5)
            #axs[5].axis('off')

            axs[4].imshow(recon_norm[ind_mvtec,...],vmin=0.35, vmax=0.5, cmap ='gist_heat')
            axs[4].set_title('Reconstruction Error', fontsize=7)
            axs[4].axis('off')

            axs[5].imshow(nln_norm[ind_mvtec,...],vmin=0.35, vmax=0.5, cmap = 'gist_heat')
            axs[5].set_title('NLN Error', fontsize=7)
            axs[5].axis('off')

            plt.tight_layout()
            path = '/tmp/residuals/{}/'.format(cls)
            print('Saved {}{}.png'.format(path, ind_mvtec))

            if not os.path.exists(path):
                os.makedirs(path)

            plt.savefig('{}{}'.format(path,ind_mvtec), dpi=300)

def get_dists(neighbours_dist, args):
    """
        Reconstruct distance vector to original dimensions when using patches

        Parameters
        ----------
        neighbours_dist (np.array): Vector of per neighbour distances
        args (Namespace): cmd_args 

        Returns
        -------
        dists (np.array): reconstructed patches if necessary

    """

    dists = np.mean(neighbours_dist, axis = tuple(range(1,neighbours_dist.ndim)))
    if args.patches:
        dists = np.array([[d]*args.patch_x**2 for i,d in enumerate(dists)]).reshape(len(dists), args.patch_x, args.patch_y)
        dists_recon = reconstruct(np.expand_dims(dists,axis=-1),args)
        return dists_recon
    else:
        return dists 

if __name__ == '__main__':
    main()
