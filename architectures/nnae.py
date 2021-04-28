import tensorflow as tf
import numpy as np
import time
from models import (Encoder, 
                   Decoder, 
                   Autoencoder)

from sklearn import neighbors
from inference import infer

from utils.plotting  import  (generate_and_save_images,
                             generate_and_save_training)

from utils.training import print_epoch,save_checkpoint
from model_config import *
from .helper import end_routine

optimizer = tf.keras.optimizers.Adam()
neighbours = 5


def nn_loss(x,x_hat,z,nn):
    x_stacked = tf.stack([x]*neighbours,axis=1)
    nle = tf.reduce_mean(tf.math.abs(tf.subtract(x_stacked,nn)))
    recon = tf.reduce_mean(tf.math.abs(tf.subtract(x,x_hat)))

    return nle + recon


#@tf.function
def train_step(model, x, nn):
    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape() as tape:
        x_hat = model(x)
        z = model.encoder(x)
        loss = nn_loss(x,x_hat,z, nn)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def train(ae,train_dataset,train_images, test_images,test_labels,args,verbose=True,save=True):
    ae_loss= []
    for epoch in range(args.epochs):
        start = time.time()
        _z = infer(ae.encoder, train_images, args, 'encoder')# ae.encoder(train_dataset)
        _x = infer(ae, train_images, args, 'AE')# ae.encoder(train_dataset)
        for image_batch in train_dataset:

            z = ae.encoder(image_batch)

            nbrs = neighbors.NearestNeighbors(n_neighbors= neighbours,
                                              algorithm='ball_tree',
                                              n_jobs=-1).fit(_z) 

            neighbours_idx =  nbrs.kneighbors(z.numpy(),return_distance=False)

            nn = tf.convert_to_tensor(_x[neighbours_idx], dtype=tf.float32)

            auto_loss  =  train_step(ae,image_batch, nn)

        generate_and_save_images(ae,
                                 epoch + 1,
                                 image_batch[:25,...],
                                 'NNAE',
                                 args)
        save_checkpoint(ae,epoch, args,'NNAE','ae')

        ae_loss.append(auto_loss)

        print_epoch('NNAE',epoch,time.time()-start,{'AE Loss':auto_loss.numpy()},None)

    generate_and_save_training([ae_loss],
                                ['ae loss'],
                                'NNAE',args)
    generate_and_save_images(ae,epoch,image_batch[:25,...],'NNAE',args)

    return ae

def main(train_dataset,train_images,train_labels,test_images,test_labels, test_masks,args):
    ae = Autoencoder(args)
    ae = train(ae,train_dataset, train_images, test_images,test_labels,args)
    end_routine(train_images, test_images, test_labels, test_masks, [ae], 'NNAE', args)

    
if __name__  == '__main__':
    main()
