import tensorflow as tf
import numpy as np
import os 
import pandas as pd
import pickle
from sklearn.metrics import roc_curve, auc, f1_score, roc_auc_score


import sys
sys.argv = [''] 

from inference import infer, get_error
from utils.data import reconstruct,process,reconstruct_latent_patches
from utils.metrics import *
from models_mvtec import Encoder as Encoder_MVTEC
from models_mvtec import Autoencoder as Autoencoder_MVTEC 
from models_mvtec import Discriminator_x as Discriminator_x_MVTEC
from models import Discriminator_z 

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
    anomaly_class='bottle',
    radius= [10],
    neighbors= [1,3, 5 ,10],
    algorithm = 'knn'
)

def main(cmd_args):
    df = pd.read_csv('outputs/results_{}_{}.csv'.format(cmd_args.data, cmd_args.seed))

    model_names = list(pd.unique(df.Name)) 
    models = list(pd.unique(df.Model))  

    for model_type in models:
        results = {}
                
        for i,clss in enumerate(list(df.Class)):
            if (('grid' in clss) or
                ('screw' in clss) or 
                ('zipper' in clss)): 
                args.set_input_shape((PATCH,PATCH,1))
            else:
                args.set_input_shape((PATCH,PATCH,3))

            if ((model_type =='VAE') and  # VAE never trained properly on these classes 
                    ((clss == 'leather') or (clss == 'tile') or (clss == 'wood') or (clss == 'zipper'))):
                continue

            detections,segmentations, n_arr, ious = [],[], [], []
            args.set_class(clss)
            (train_dataset, train_images, train_labels, test_images, test_labels,test_masks) = load_mvtec(args)


            model_name = model_names[i]
            args.set_name(model_name)
            ae = Autoencoder_MVTEC(args)
            p = 'outputs/{}/{}/{}/training_checkpoints/checkpoint_full_model_ae'.format(model_type, clss, model_name)
            ae.load_weights(p)

            if model_type == 'GANomaly':
                encoder = tf.keras.Sequential(Encoder_MVTEC(args))
                p = 'outputs/{}/{}/{}/training_checkpoints/checkpoint_full_model_encoder'.format(model_type, clss, model_name)
                encoder.load_weights(p)
                model = [ae,None,encoder]

            elif model_type == 'DAE_disc':
                disc = Discriminator_x_MVTEC(args)
                p = 'outputs/{}/{}/{}/training_checkpoints/checkpoint_full_model_disc'.format(model_type, clss, model_name)
                disc.load_weights(p)
                model = [ae,disc]

            else: model = [ae]
            
            x_hat  = infer(model[0], test_images, args, 'AE')
            x_hat_train  = infer(model[0], train_images, args, 'AE')
            z_query = infer(model[0].encoder, test_images, args, 'encoder') 
            z = infer(model[0].encoder, train_images, args, 'encoder')

            error = get_error('AE', test_images, x_hat, mean=False) 

            for nneigh in args.neighbors:
                neighbours_dist, neighbours_idx, x_hat_train, neighbour_mask = nln(z, 
                                                                                   z_query, 
                                                                                   x_hat_train, 
                                                                                   args.algorithm, 
                                                                                   nneigh, 
                                                                                   radius=None)
                nln_error = get_nln_errors(model,
                                           model_type,
                                           z_query,
                                           z,
                                           test_images,
                                           x_hat_train,
                                           neighbours_idx,
                                           neighbour_mask,
                                           args)

                nln_error_seg = get_nln_errors([ae],
                                           'AE',
                                           z_query,
                                           z,
                                           test_images,
                                           x_hat_train,
                                           neighbours_idx,
                                           neighbour_mask,
                                           args)


                nln_error_seg_recon,test_labels_recon = reconstruct(nln_error_seg, args,test_labels)
                if model_type == 'DAE_disc'  or model_type  == 'GANomaly': 
                    nln_error_recon,test_labels_recon = reconstruct_latent_patches(nln_error, args,test_labels)
                else:
                    nln_error_recon,test_labels_recon = reconstruct(nln_error, args,test_labels)
                error_recon = reconstruct(error, args)
                masks_recon= reconstruct(np.expand_dims(test_masks,axis=-1), args)[...,0]
                dists_recon = get_dists(neighbours_dist, args)

                ####### DETECTION
                nln_norm = process(nln_error_recon, per_image=False)
                recon_norm = process(error_recon, per_image=False)
                dists_norm = process(dists_recon, per_image=False)

                error_agg = aggregate(recon_norm,method='max') 
                nln_error_agg = aggregate(nln_norm, method='max')
                dists_agg = aggregate(dists_norm, method='max')

                alpha = 0.5 
                add_norm = (1-alpha)*nln_error_agg + alpha*dists_agg
                AUC_detection = roc_auc_score(test_labels_recon == args.anomaly_class, add_norm.flatten())
                detections.append(AUC_detection)

                ####### SEGMENTATION
                nln_norm = process(nln_error_seg_recon, per_image=False)
                
                alpha = 0.5 
                add_norm =  (1-alpha)*np.mean(nln_norm,axis=-1) + alpha*dists_norm[...,0]
                AUC_segmentation = roc_auc_score(masks_recon.flatten()>0, add_norm.flatten())
                segmentations.append(AUC_segmentation)

                ###### IOU
                ious.append(iou_score(add_norm, masks_recon))
                n_arr.append(nneigh)
                results[clss]  = {'neighbour': n_arr,
                                   'detection':detections,
                                   'segmentation':segmentations,
                                   'iou': ious}
                print(results[clss])
                filename = 'outputs/{}_{}_{}.pkl'.format(args.data, model_type,cmd_args.seed)
                with open(filename,'wb') as f:
                    pickle.dump(results,f)
    find_best(models,cmd_args.seed)

def aggregate(xs, method='avg'):
    y = np.empty(xs.shape[0])
    if method =='avg':
        for i,x in enumerate(xs):
            y[i] = np.mean(x)
    elif method == 'max':
        for i,x in enumerate(xs):
            y[i] = np.max(x)
    elif method == 'med':
        for i,x in enumerate(xs):
            y[i] = np.median(x)
    return y

def find_best(models,seed):
    results = {}
    for model_type in models:
        filename = 'outputs/{}_{}_{}.pkl'.format(args.data, model_type,seed)
        d = np.load(filename, allow_pickle=True)
        df = pd.DataFrame(columns = ['class', 'neighbour', 'detection', 'segmentation'])
        for key in d.keys():
            df_temp = pd.DataFrame(d[key], columns =  ['class', 'neighbour', 'detection', 'segmentation'])
            df_temp['class'] = key
            df = df.append(df_temp)

        df_group = df.groupby(['neighbour']).agg({'segmentation':'mean','detection':'mean'}).reset_index()

        results[model_type] = [df_group.segmentation.max(), df_group.detection.max()]

        if model_type == 'GANomaly':
            print('')
            idx = df_group.segmentation.idxmax()
            n = df_group.iloc[idx].neighbour
            df_max = df[df.neighbour ==n].reset_index()
            df_max = df_max.drop(['index'], axis=1)
            df_max = df_max.round(2)

            print(df_max)

            textures = df_max[(df_max['class'] == 'carpet') | 
                              (df_max['class'] == 'grid') | 
                              (df_max['class'] == 'leather') | 
                              (df_max['class'] == 'tile') | 
                              (df_max['class'] == 'wood')]

            objects =  df_max[(df_max['class'] != 'carpet') & 
                              (df_max['class'] != 'grid') & 
                              (df_max['class'] != 'leather') & 
                              (df_max['class'] != 'tile') & 
                              (df_max['class'] != 'wood')]
            print('Textures')
            print(textures)
            print(textures.mean())
            print('Objects')
            print(objects)
            print(objects.mean())

            print('')

    print(results)
                



if __name__ == '__main__':
    main()
