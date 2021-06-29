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
from utils.metrics.mvtec_metrics import get_dists
from models_mvtec import Encoder as Encoder_MVTEC
from models_mvtec import Autoencoder as Autoencoder_MVTEC 
from models_mvtec import Discriminator_x as Discriminator_x_MVTEC

from data import *
from inference import *

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    def set_class(self,clss):
        self.anomaly_class = clss
    def set_name(self,clss):
        self.model_name= clss
    def set_input_shape(self,input_shape):
        self.input_shape= input_shape 

PATCH = 128 
LD = 128

args = Namespace(
    data='MVTEC',
    seed='12334',
    input_shape=(PATCH, PATCH, 3),
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
    latent_dim=LD,
    # NLN PARAMS
    anomaly_class='????',
    radius= [10],
    neighbors= [5],
    algorithm = 'knn'
)


def main(cmd_args):
    df = pd.read_csv('outputs/results_{}_{}.csv'.format(cmd_args.data, cmd_args.seed)) 
    dataset = 'MVTEC'
    df = df[df.Model == 'AE']
    for cls in list(df.Class):
        if (('grid' in cls) or
            ('screw' in cls) or 
            ('zipper' in cls)): 
            args.set_input_shape((PATCH,PATCH,1))
        else:
            args.set_input_shape((PATCH,PATCH,3))

        args.set_class(cls)
        (train_dataset, train_images, train_labels, test_images, test_labels,test_masks) = load_mvtec(args)

        ae = Autoencoder_MVTEC(args)        

        ae.load_weights('/home/mmesarcik/NLN/outputs/AE/{}/{}/training_checkpoints/checkpoint_full_model_ae'.format(cls,names[cls]))

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

        test_images_recon ,test_labels_recon = reconstruct(test_images, args,test_labels)
        nln_error_recon = reconstruct(nln_error, args)
        error_recon = reconstruct(error, args)
        masks_recon= reconstruct(np.expand_dims(test_masks,axis=-1), args)[...,0]
        dists_recon = get_dists(neighbours_dist, args)

        nln_norm = process(nln_error_recon, per_image=False)
        recon_norm = process(error_recon, per_image=False)
        dists_norm = process(dists_recon, per_image=False)
        add_norm =  0.5*np.mean(nln_norm,axis=-1) + 0.5*dists_norm[...,0]
        
        for r in np.where(test_labels_recon == cls)[0]:

            fig, axs  = plt.subplots(1,3,figsize=(9,3))
            
            if (('grid' in cls) or
                ('screw' in cls) or 
                ('zipper' in cls)): 
                axs[0].imshow(test_images_recon[r,...,0],cmap='gray'); axs[0].set_title('Input',fontsize=15)
            else:
                axs[0].imshow(test_images_recon[r]); axs[0].set_title('Input', fontsize=15)

            axs[1].imshow(masks_recon[r],cmap='gist_heat'); axs[1].set_title('Ground Truth', fontsize=15)
            axs[2].imshow(add_norm[r], vmin=0.25, vmax=0.5, cmap='gist_heat'); axs[2].set_title('NLN Output',fontsize=15)

            axs[0].axis('off'); axs[1].axis('off'); axs[2].axis('off')

            path = '/tmp/segmentation/{}/'.format(cls)
            print('Saved {}/{}.png'.format(path, r))

            if not os.path.exists(path):
                os.makedirs(path)

            plt.savefig('{}{}'.format(path,r), dpi=300)

if __name__ == '__main__':
    main()
