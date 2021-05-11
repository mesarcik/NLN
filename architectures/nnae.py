import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import time
from models import (MultiEncoder, 
                   Autoencoder)

from sklearn import neighbors
from inference import infer

from utils.plotting  import  (generate_and_save_images,
                             generate_and_save_training)

from utils.training import print_epoch,save_checkpoint
from model_config import *
from .helper import end_routine

ae_optimizer = tf.keras.optimizers.Adam()
encoder_optimizer = tf.keras.optimizers.Adam()
NNEIGHBOURS= 5


def nn_loss(x,x_hat,neigh):
    
    x_stack = tf.stack([x]*NNEIGHBOURS,axis=1)
    
    nle = tf.reduce_mean(tf.math.abs(tf.subtract(x_stack,neigh))) 
    recon = tf.reduce_mean(tf.math.abs(tf.subtract(x,x_hat))) 

    return nle + recon


def l2_loss(x,x_hat):
    return tf.reduce_mean(tf.math.abs(tf.subtract(x,x_hat)),name='l2loss')

def z_loss(z, z_hat):
    return tf.reduce_mean(tf.math.abs(tf.subtract(z,z_hat)),name='l2loss')

@tf.function
def train_step(ae, encoder, x, neighs):
    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape() as ae_tape, tf.GradientTape() as encoder_tape:
        x_hat = ae(x)
        z = ae.encoder(x)
        z_hat = encoder(x_hat, [neighs[:,i,...] for i in range(neighs.shape[1])])

        ae_loss = l2_loss(x,x_hat)
        encoder_loss = z_loss(z,z_hat) 

    ae_gradients = ae_tape.gradient(ae_loss, ae.trainable_variables)
    encoder_gradients = encoder_tape.gradient(encoder_loss, encoder.trainable_variables)

    ae_optimizer.apply_gradients(zip(ae_gradients, ae.trainable_variables))
    encoder_optimizer.apply_gradients(zip(encoder_gradients, encoder.trainable_variables))
    return ae_loss, encoder_loss

def train(ae, encoder,train_dataset, train_images, test_images,test_labels,args,verbose=True,save=True):
    ae_loss, e_loss= [], []
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).batch(BATCH_SIZE)
    for epoch in range(args.epochs):
        start = time.time()

        _z = infer(ae.encoder, train_images, args, 'encoder')
        _x = infer(ae , train_images, args, 'AE')
        nbrs = neighbors.NearestNeighbors(n_neighbors= NNEIGHBOURS, algorithm='ball_tree', n_jobs=-1).fit(_z) 

        neighbours_dist, neighbours_idx  =  nbrs.kneighbors(_z, return_distance=True)
        neighbours = _x[neighbours_idx]

        ############ DELETE 
        fig,axs = plt.subplots(5,NNEIGHBOURS +2,figsize=(5,5))
        for i in range(5):
            r = np.random.randint(train_images.shape[0])
            axs[i,0].imshow(train_images[r,...])
            axs[i,1].imshow(_x[r,...])
            for j in range(2,NNEIGHBOURS+2):
                axs[i,j].imshow(neighbours[r,j-2,...])
                axs[i,j].set_title('N{} - {}'.format(j-1, round(neighbours_dist[r,j-2]),3),fontsize=5)
                axs[i,j].axis('off')
        plt.savefig('/tmp/neighbours/n_{}_{}'.format(args.anomaly_class,epoch))
        plt.close('all')
        ###################

        neighbours = tf.convert_to_tensor(neighbours, dtype=tf.float32)
        neighbours_dataet = tf.data.Dataset.from_tensor_slices(neighbours).batch(BATCH_SIZE)

        for image_batch, neighs in zip(train_dataset,neighbours_dataet):
            auto_loss, encoder_loss  =  train_step(ae, encoder,image_batch, neighs)

        generate_and_save_images(ae,
                                 epoch + 1,
                                 image_batch[:25,...],
                                 'NNAE',
                                 args)

        save_checkpoint(ae,epoch, args,'NNAE','ae')
        save_checkpoint(encoder,epoch, args,'NNAE','encoder')

        ae_loss.append(auto_loss)
        e_loss.append(encoder_loss)

        print_epoch('NNAE',epoch,time.time()-start,{'AE Loss':auto_loss.numpy(),
                                                    'Encoder Loss': encoder_loss.numpy()},None)

    generate_and_save_training([ae_loss,e_loss],
                                ['ae loss', 'encoder loss'],
                                'NNAE',args)
    generate_and_save_images(ae,epoch,image_batch[:25,...],'NNAE',args)

    return ae

def main(train_dataset,train_images,train_labels,test_images,test_labels, test_masks,args):
    ae = Autoencoder(args)
    encoder = MultiEncoder(args)
    ae = train(ae, encoder, train_dataset, train_images, test_images,test_labels,args)
    end_routine(train_images, test_images, test_labels, test_masks, [ae, encoder], 'NNAE', args)

    
if __name__  == '__main__':
    main()
