import tensorflow as tf
import numpy as np
import time
from models import (Encoder_new, 
                   Autoencoder_new)

from utils.plotting  import  (generate_and_save_images,
                             generate_and_save_training,
                             save_training_curves)

from utils.training import print_epoch,save_checkpoint
from utils.metrics import get_classifcation,nearest_error,save_metrics
from model_config import *

optimizer = tf.keras.optimizers.Adam(1e-5)

def l2_loss(x,x_hat):
    return cross_entropy(x,x_hat)

@tf.function
def train_step(model, x,y):
    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape() as tape:
        x_hat = model(x,y)
        loss = l2_loss(x,x_hat)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def train(ae,train_dataset,train_dataset_labels, test_images,test_labels,args,verbose=True,save=True):
    ae_loss,aucs = [], []
    for epoch in range(args.epochs):
        start = time.time()
        for image_batch,label_batch in zip(train_dataset,train_dataset_labels):
            auto_loss  =  train_step(ae,image_batch,label_batch)

        generate_and_save_images(ae,
                                 epoch + 1,
                                 image_batch[:25,...],
                                 label_batch[:25,...],
                                 'AE_labels',
                                 args)
        save_checkpoint(ae,epoch, args,'AE_labels','ae')

        ae_loss.append(auto_loss)
        roc_auc,f1 = get_classifcation('AE_labels',
                                        ae,
                                       test_images,
                                       test_labels,
                                       args.anomaly_class,
                                       hera=args.data=='HERA')
        aucs.append(roc_auc)

        print_epoch('AE_labels',epoch,time.time()-start,{'AE Loss':auto_loss.numpy()},roc_auc)

    generate_and_save_training([ae_loss,aucs],
                                ['ae loss','AUC'],
                                'AE_labels',args)
    generate_and_save_images(ae,epoch,image_batch[:25,...],label_batch[:25,...],'AE_labels',args)

    return ae

def main(train_dataset,train_dataset_labels,train_images,train_labels,test_images,test_labels,args):
    ae = Autoencoder_new(args)
    ae = train(ae,train_dataset,train_dataset_labels,test_images,test_labels,args)
    save_training_curves(ae,args,test_images,test_labels,'AE_labels')
    auc_latent, f1_latent, neighbour,radius = nearest_error(ae,
                                                     train_images,
                                                     test_images,
                                                     test_labels,
                                                     'AE_labels',
                                                     args,
                                                     args.data == 'HERA')
    
    auc_recon ,f1_recon = get_classifcation('AE_labels',
                                             ae,
                                             test_images,
                                             test_labels,
                                             args.anomaly_class,
                                             hera = args.data == 'HERA',
                                             f1=True)
    save_metrics('AE_labels',
                 args.model_name,
                 args.data,
                 args.anomaly_class,
                 auc_recon, 
                 f1_recon,
                 neighbour,
                 radius,
                 auc_latent,
                 f1_latent)
    
if __name__  == '__main__':
    main()
