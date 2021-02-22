import tensorflow as tf
import numpy as np
import time
from models import (Encoder, 
                   Decoder, 
                   Autoencoder)

from utils.plotting  import  (generate_and_save_images,
                             generate_and_save_training,
                             save_training_curves)

from utils.training import print_epoch,save_checkpoint
from utils.metrics import get_classifcation,nearest_error,save_metrics
from model_config import *

optimizer = tf.keras.optimizers.Adam()

def l2_loss(x,x_hat):

    return tf.reduce_mean(tf.math.abs(tf.subtract(x,x_hat)))


@tf.function
def train_step(model, x):
    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape() as tape:
        x_hat = model(x)
        loss = l2_loss(x,x_hat)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def train(ae,train_dataset,test_images,test_labels,args,verbose=True,save=True):
    ae_loss,aucs = [], []
    for epoch in range(args.epochs):
        start = time.time()
        for image_batch in train_dataset:
           auto_loss  =  train_step(ae,image_batch)

        generate_and_save_images(ae,
                                 epoch + 1,
                                 image_batch[:25,...],
                                 'AE',
                                 args)
        save_checkpoint(ae,epoch, args,'AE','ae')

        ae_loss.append(auto_loss)
        roc_auc,f1 = get_classifcation('AE',
                                        ae,
                                       test_images,
                                       test_labels,
                                       args.anomaly_class,
                                       hera=args.data=='HERA')
        aucs.append(roc_auc)

        print_epoch('AE',epoch,time.time()-start,{'AE Loss':auto_loss.numpy()},roc_auc)

    generate_and_save_training([ae_loss,aucs],
                                ['ae loss','AUC'],
                                'AE',args)
    generate_and_save_images(ae,epoch,image_batch[:25,...],'AE',args)

    return ae

def main(train_dataset,train_images,train_labels,test_images,test_labels,args):
    ae = Autoencoder(args)
    ae = train(ae,train_dataset,test_images,test_labels,args)
    save_training_curves(ae,args,test_images,test_labels,'AE')
    auc_latent, f1_latent, neighbour,radius = nearest_error(ae,
                                                     train_images,
                                                     test_images,
                                                     test_labels,
                                                     'AE',
                                                     args,
                                                     args.data == 'HERA')
    
    auc_recon ,f1_recon = get_classifcation('AE',
                                             ae,
                                             test_images,
                                             test_labels,
                                             args.anomaly_class,
                                             hera = args.data == 'HERA',
                                             f1=True)
    save_metrics('AE',
                 args,
                 auc_recon, 
                 f1_recon,
                 neighbour,
                 radius,
                 auc_latent,
                 f1_latent)
    
if __name__  == '__main__':
    main()
