import tensorflow as tf
import numpy as np
import time
from models import (Encoder, 
                   Decoder, 
                   Autoencoder)

from sklearn import neighbors

from utils.plotting  import  (generate_and_save_images,
                             generate_and_save_training)

from utils.training import print_epoch,save_checkpoint
from model_config import *
from .helper import end_routine

optimizer = tf.keras.optimizers.Adam()
neighbours = 3

def nn_loss(x,x_hat,z,_z,nbrs):

    neighbours_dist, neighbours_idx =  nbrs.kneighbors(z_query,return_distance=True)#KNN
    z_stacked = tf.stack([test_images]*neighbours,axis=1)
    nle = tf.math.abs(tf.subtract(z_stacked,_z[neighbors]))

    return tf.reduce_mean(nle)


@tf.function
def train_step(model, x, _z, nbrs):
    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape() as tape:
        x_hat = model(x)
        z = model.encoder(x)
        loss = nn_loss(x,x_hat,z, _z, nbrs)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def train(ae,train_dataset,test_images,test_labels,args,verbose=True,save=True):
    ae_loss= []
    for epoch in range(args.epochs):
        start = time.time()
        for image_batch in train_dataset:
            _z = ae.encoder(image_batch)
            nbrs = neighbors.NearestNeighbors(n_neighbors= neighbours,
                                              algorithm='ball_tree',
                                              n_jobs=-1).fit(_z) 
            auto_loss  =  train_step(ae,image_batch, _z, nbrs)

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
    ae = train(ae,train_dataset,test_images,test_labels,args)
    end_routine(train_images, test_images, test_labels, test_masks, [ae], 'NNAE', args)

    
if __name__  == '__main__':
    main()
