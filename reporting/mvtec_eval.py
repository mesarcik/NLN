import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import roc_curve, auc, f1_score, roc_auc_score, jaccard_score 
from matplotlib import pyplot as plt

import sys
sys.argv = [''] 
sys.path.insert(1,'/home/mmesarcik/NLN/')

from inference import infer, get_error
from data import *
from utils.data import reconstruct,process,reconstruct_latent_patches
from utils.metrics import  nln, get_nln_errors
from models_mvtec import Encoder as Encoder_MVTEC
from models_mvtec import Autoencoder as Autoencoder_MVTEC 
from models_mvtec import Discriminator_x as Discriminator_x_MVTEC
from models import Discriminator_z 



class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    def set_class(self,clss):
        self.anomaly_class = clss
    def set_name(self,clss):
        self.model_name= clss
    def set_input_shape(self,input_shape):
        self.input_shape= input_shape 
    def set_mvtec_path(self,path):
        self.mvtec_path = path

PATCH = 128 
LD = 128


args = Namespace(
    data='MVTEC',
    mvtec_path = '/data/mmesarcik/MVTecAD/',
    seed='final',
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
    anomaly_class='tba',
    radius= [10],
    neighbors= [1, 2, 5, 10],
    algorithm = 'knn'
)

def main(cmd_args):
    df = pd.read_csv('outputs/results_{}_{}.csv'.format(cmd_args.data, cmd_args.seed))
    df_out = pd.DataFrame(columns=['Model','Dataset','Name','Latent_Dim', 'Patch_Size', 
                                  'Class', 'Type', 'Neighbour', 'Alpha', 
                                  'Det_Recon_AUROC', 'Det_Add_AUROC',
                                  'Seg_Recon_AUROC', 'Seg_Add_AUROC',
                                  'Seg_Recon_IOU', 'Seg_Logical_And_IOU', 'Seg_Add_IOU'])
    df_index = 0

    models = list(pd.unique(df.Model))  

    for model_type in models:
        results = {}
                
        for i,clss in enumerate(list(pd.unique(df.Class))):
            if (('grid' in clss) or
                ('screw' in clss) or 
                ('zipper' in clss)): 
                args.set_input_shape((PATCH,PATCH,1))
            else:
                args.set_input_shape((PATCH,PATCH,3))

            model_name = df[(df.Model == model_type) &
                            (df.Class == clss)].Name
            if model_name.empty: continue
            else: model_name = model_name.iloc[0]

            detections,segmentations, n_arr, ious = [],[], [], []
            args.set_class(clss)
            args.set_mvtec_path(cmd_args.mvtec_path)
            (train_dataset, train_images, train_labels, test_images, test_labels,test_masks) = load_mvtec(args)


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

                error_agg     = np.nanmean(recon_norm,axis=tuple(range(1,recon_norm.ndim))) 
                nln_error_agg = np.nanmean(nln_norm,axis=tuple(range(1,nln_norm.ndim))) 
                dists_agg     = np.max(dists_norm,axis=tuple(range(1,dists_norm.ndim)))

                det_recon_auroc = roc_auc_score(test_labels_recon == args.anomaly_class, error_agg.flatten())

                ####### SEGMENTATION
                nln_norm = process(nln_error_seg_recon, per_image=False)
                seg_recon_auroc = roc_auc_score(masks_recon.flatten()>0, np.nanmean(recon_norm,axis=-1).flatten())
                seg_recon_iou = iou_score(np.nanmean(recon_norm, axis=-1), masks_recon) 
                seg_logical_and_iou = iou_score(np.nanmean(nln_norm, axis=-1), masks_recon, dists=np.max(dists_norm, axis=-1)) 


                for alpha in [0, 0.20, 0.4, 0.6, 0.8, 1.00]:
                    
                    error_det = (1-alpha)*nln_error_agg + alpha*dists_agg
                    error_seg = (1-alpha)*np.nanmean(nln_norm, axis=-1) + alpha*np.nanmean(dists_norm, axis=-1)

                    det_add_auroc = roc_auc_score(test_labels_recon == args.anomaly_class, error_det.flatten())
                    seg_add_auroc = roc_auc_score(masks_recon.flatten()>0, error_seg.flatten())

                    seg_add_iou= iou_score(error_seg, masks_recon)

                    df_out.loc[df_index] = [model_type,'MVTEC',model_name, LD, PATCH, 
                                            clss, 'SIMO', nneigh, alpha, 
                                            det_recon_auroc, det_add_auroc,
                                            seg_recon_auroc, seg_add_auroc, 
                                            seg_recon_iou, seg_logical_and_iou,  seg_add_iou]

                    filename = 'outputs/results_{}_{}_{}.csv'.format(args.data, cmd_args.seed, 'joined')
                    df_out.to_csv(filename, index=False)
                    df_index+=1

    #find_best(models,cmd_args.seed)

def aggregate(xs, method='mean'):
    y = np.empty(xs.shape[0])
    if method =='mean':
        for i,x in enumerate(xs):
            y[i] = np.nanmean(x)
    elif method == 'mean':
        for i,x in enumerate(xs):
            y[i] = np.max(x)
    elif method == 'med':
        for i,x in enumerate(xs):
            y[i] = np.median(x)
    return y

def find_best():
    results = {}
    filename = 'outputs/results_{}_{}_{}.csv'.format(args.data, args.seed, 'joined')
    df = pd.read_csv(filename)
    fig, axs = plt.subplots(1,3,figsize=(9,3))

    for model in list(pd.unique(df.Model)):
        SEG_AUROC, SEG_LD, SEG_N, SEG_A, seg_aurocs = 0, 0 ,0, 0, [0,0]
        DET_AUROC, DET_LD, DET_N, DET_A, det_aurocs = 0, 0 ,0, 0, [0,0]
        IOU, IOU_LD, IOU_N, IOU_A, ious = 0, 0 ,0, 0, [0,0]
        for ld in list(pd.unique(df.Latent_Dim)):
            for neigh in list(pd.unique(df.Neighbour)):
                seg_add_aurocs_temp = []
                det_add_aurocs_temp = []
                ious_temp= []
                for alpha in list(pd.unique(df.Alpha)):
                    df_temp = df[((df.Model == model) & (df.Latent_Dim == ld) & (df.Neighbour == neigh) & (df.Alpha == alpha))]
                    seg_add_auroc_temp =  df_temp['Seg_Add_AUROC'].mean()
                    det_add_auroc_temp =  df_temp['Det_Add_AUROC'].mean()
                    iou_temp =  df_temp['Seg_Add_IOU'].mean()

                    seg_add_aurocs_temp.append(seg_add_auroc_temp)
                    det_add_aurocs_temp.append(det_add_auroc_temp)
                    ious_temp.append(iou_temp)

                    if round(seg_add_auroc_temp,3) > SEG_AUROC:
                        SEG_AUROC, SEG_LD, SEG_N, SEG_A  = round(seg_add_auroc_temp,3), ld, neigh, alpha
                    if round(det_add_auroc_temp,3) > DET_AUROC:
                        DET_AUROC, DET_LD, DET_N, DET_A  = round(det_add_auroc_temp,3), ld, neigh, alpha
                    if round(iou_temp,3) > IOU:
                        IOU, IOU_LD, IOU_N, IOU_A  = round(iou_temp,3), ld, neigh, alpha

                if np.mean(seg_add_aurocs_temp) > np.mean(seg_aurocs): seg_aurocs = seg_add_aurocs_temp
                if np.mean(det_add_aurocs_temp) > np.mean(det_aurocs): det_aurocs = det_add_aurocs_temp
                if np.mean(ious_temp) > np.mean(ious): ious = ious_temp 

        axs[0].plot(list(pd.unique(df.Alpha)), seg_aurocs, marker='o', label = ('{}'.format(model)))
        axs[0].grid(True)
        axs[0].legend()
        axs[0].set_title('Pixel-based AUROC vs Alpha',fontsize=10)
        axs[0].set_xlabel(r'$\alpha$')
        axs[0].set_ylabel('AUROC')

        axs[1].plot(list(pd.unique(df.Alpha)), det_aurocs, marker='o', label = ('{}'.format(model)))
        axs[1].grid(True)
        axs[1].legend()
        axs[1].set_title('Image-based AUROC vs Alpha',fontsize=10)
        axs[1].set_xlabel(r'$\alpha$')
        axs[1].set_ylabel('AUROC')

        axs[2].plot(list(pd.unique(df.Alpha)), ious, marker='o',label = ('{}'.format(model)))
        axs[2].grid(True)
        axs[2].legend()
        axs[2].set_title('IOU vs Alpha',fontsize=10)
        axs[2].set_xlabel(r'$\alpha$')
        axs[2].set_ylabel('IoU')

        # For each model print the per class scores:
        print('_______\n{}\n_______'.format(model))
        print('_______\n{}\n_______'.format('Segmentation'))
        print(df[(df.Model== model) & 
                 (df.Latent_Dim == SEG_LD) &
                 (df.Neighbour == SEG_N) &
                 (df.Alpha == SEG_A)])
        print('_______\n{}\n_______'.format('Detection'))
        print(df[(df.Model== model) & 
                 (df.Latent_Dim == DET_LD) &
                 (df.Neighbour == DET_N) &
                 (df.Alpha == DET_A)])
        print('_______\n{}\n_______'.format('IOU'))
        print(df[(df.Model== model) & 
                 (df.Latent_Dim == IOU_LD) &
                 (df.Neighbour == IOU_N) &
                 (df.Alpha == IOU_A)])
        print('_______')

    plt.tight_layout()
    plt.savefig('/tmp/temp',dpi=300)

    #df_seg = (df[(df.Model== model) & 
    #         (df.Latent_Dim == SEG_LD) &
    #         (df.Neighbour == SEG_N) &
    #         (df.Alpha == SEG_A)])

    #print('_______')
    #print('_______')
    #textures = df_seg[(df_seg['Class'] == 'carpet') | 
    #                  (df_seg['Class'] == 'grid') | 
    #                  (df_seg['Class'] == 'leather') | 
    #                  (df_seg['Class'] == 'tile') | 
    #                  (df_seg['Class'] == 'wood')]['Seg_Add_AUROC']

    #objects =  df_seg[(df_seg['Class'] != 'carpet') & 
    #                  (df_seg['Class'] != 'grid') & 
    #                  (df_seg['Class'] != 'leather') & 
    #                  (df_seg['Class'] != 'tile') & 
    #                  (df_seg['Class'] != 'wood')]['Seg_Add_AUROC']
    #print('Textures')
    #print(textures)
    #print(textures.mean())
    #print('Objects')
    #print(objects)
    #print(objects.mean())

    #print('')

    #print(results)
                

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

    dists = np.max(neighbours_dist, axis = tuple(range(1,neighbours_dist.ndim)))
    if args.patches:
        dists = np.array([[d]*args.patch_x**2 for i,d in enumerate(dists)]).reshape(len(dists), args.patch_x, args.patch_y)
        dists_recon = reconstruct(np.expand_dims(dists,axis=-1),args)
        return dists_recon
    else:
        return dists 

def iou_score(error, test_masks, dists=None):
    """
        Get jaccard index or IOU score

        Parameters
        ----------
        error (np.array): input-output
        test_masks (np.array): ground truth mask 

        Returns
        -------
        max_iou (float32): maximum iou score for a number of thresholds

    """
    fpr,tpr, thr = roc_curve(test_masks.flatten()>0, error.flatten())
    thr = thr[np.argmax(tpr-fpr)]
    thresholded = error >=thr
    if dists is not None:
        fpr_d,tpr_d, thr_d = roc_curve(test_masks.flatten()>0, dists.flatten())
        thr_d = thr_d[np.argmax(tpr_d-fpr_d)]
        thresholded_d = dists >=thr_d
        return jaccard_score(test_masks.flatten()>0, np.logical_and(thresholded, thresholded_d).flatten())
    else:
        return jaccard_score(test_masks.flatten()>0, thresholded.flatten())

if __name__ == '__main__':
    main(args)
