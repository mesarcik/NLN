import tensorflow as tf
import numpy as np
import os 
import pandas as pd
from sklearn.metrics import roc_curve, auc, f1_score
from math import isnan

import sys
sys.argv = [''] 

from inference import infer, get_error
from utils.data import reconstruct
from utils.metrics import *
from models import *

from data import *
from inference import *

from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square

MODEL_PATH = '/data/mmesarcik/old_models/AE'
MODEL_NAMES = [
    'crouching-provocative-salmon-from-camelot',
    #'robust-optimal-partridge-from-tartarus',   
    #'cheerful-charming-skink-of-spirit',
    #'nifty-steel-rook-of-justice',              
    #'warping-maroon-raven-of-might',            
    #'astute-dangerous-scorpion-of-glee',        
    #'piquant-tangerine-kangaroo-of-freedom',    
    #'chubby-super-quokka-of-serendipity',       
    #'frisky-enigmatic-jacamar-of-storm',        
    #'adaptable-macho-grebe-from-nibiru',        
    #'daring-honest-robin-of-economy',           
    #'famous-burrowing-squirrel-from-pluto',     
    #'godlike-pumpkin-goose-of-unity',           
    #'hissing-adorable-aardwark-of-foundation',  
    #'exuberant-fuzzy-marmoset-from-shambhala',  
    #'arrogant-gleaming-aardwolf-of-triumph'
    ]    
classes = [
    #'bottle',
    'cable',
    #'capsule',
    #'carpet',
    #'grid',
    #'hazelnut',
    #'leather',
    #'metal_nut',
    #'pill',
    #'screw', 
    #'tile', 
    #'toothbrush',
    #'transistor', 
    #'wood',
    #'zipper'
    ]

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
    limit= 10,
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
    neighbors= [5],
    algorithm = 'knn'
)

def main():
    for i in range(15):
        if (('grid' in classes[i]) or
            ('screw' in classes[i]) or 
            ('zipper' in classes[i])): 
            args.set_input_shape((PATCH,PATCH,1))
        else:
            args.set_input_shape((PATCH,PATCH,3))

        ae = Autoencoder(args)

        args.set_class(classes[i])
        args.set_name(MODEL_NAMES[i])
        p = '{}/{}/{}/training_checkpoints/checkpoint_full_model_ae'.format(MODEL_PATH, classes[i], MODEL_NAMES[i])
        print(p)
        ae.load_weights(p)
        (train_dataset, train_images, train_labels, test_images, test_labels,test_masks) = load_mvtec(args)
        
        #x_hat  = infer(ae, test_images, args, 'AE')
        #error = get_error('AE', test_images, x_hat, mean=False) 
        
        auc_latent, f1_latent, neighbour,radius = get_nln_metrics([ae],
                                                                  train_images,
                                                                  test_images,
                                                                  test_labels,
                                                                  'AE',
                                                                  args)

        auc_recon ,f1_recon = get_classifcation('AE',
                                                [ae],
                                                test_images,
                                                test_labels,
                                                args,
                                                f1=True)

        seg_auc, seg_auc_nln, dists_auc  = accuracy_metrics([ae],
                                                            train_images,
                                                            test_images,
                                                            test_labels,
                                                            test_masks,
                                                            'AE',
                                                            neighbour,
                                                            radius,
                                                            args)

        save_metrics('AE',
                     args,
                     auc_recon, 
                     f1_recon,
                     neighbour,
                     radius,
                     auc_latent,
                     f1_latent,
                     seg_auc,
                     seg_auc_nln,
                     dists_auc)

    #fpr, tpr, thr  = roc_curve(test_masks.flatten()>0, areas)



if __name__ == '__main__':
    main()
