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

ae_optimizer = tf.keras.optimizers.Adam()
enc_optimizer = tf.keras.optimizers.Adam()
neighbours = 5


def nn_loss(x,x_hat,dists):
    nle = tf.reduce_mean(dists)
    recon = 0*tf.reduce_mean(tf.math.abs(tf.subtract(x,x_hat))) # a hack to make tf work ?????

    return nle + recon


def l2_loss(x,x_hat):
    return tf.reduce_mean(tf.math.abs(tf.subtract(x,x_hat)),name='l2loss')

def z_loss(dists):
    return tf.reduce_mean(dists,name='encloss')

@tf.function
def train_step(model, x, dists):
    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape() as ae_tape, tf.GradientTape() as enc_tape:
        x_hat = model(x)
        ae_loss = l2_loss(x,x_hat)
        enc_loss = nn_loss(x,x_hat,dists)
    ae_gradients = ae_tape.gradient(ae_loss, model.trainable_variables)
    enc_gradients = enc_tape.gradient(enc_loss, model.encoder.trainable_variables)

    ae_optimizer.apply_gradients(zip(ae_gradients, model.trainable_variables))
    enc_optimizer.apply_gradients(zip(enc_gradients, model.encoder.trainable_variables))
    return ae_loss, enc_loss

def train(ae,train_dataset,train_images, test_images,test_labels,args,verbose=True,save=True):
    ae_loss, enc_loss = [], []
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).batch(BATCH_SIZE)
    for epoch in range(args.epochs):
        start = time.time()
        _z = infer(ae.encoder, train_images, args, 'encoder')# ae.encoder(train_dataset)
        nbrs = neighbors.NearestNeighbors(n_neighbors= neighbours, algorithm='ball_tree', n_jobs=-1).fit(_z) 
        z = ae.encoder(test_images)

        neighbours_dist, _ =  nbrs.kneighbors(z.numpy(), return_distance=True)
        dists = tf.convert_to_tensor(neighbours_dist, dtype=tf.float32)

        dists = tf.data.Dataset.from_tensor_slices(dists).batch(BATCH_SIZE)

        for image_batch,dist in zip(train_dataset,dists):
            auto_loss, encoder_loss =  train_step(ae,image_batch, dist)

        generate_and_save_images(ae,
                                 epoch + 1,
                                 image_batch[:25,...],
                                 'NNAE',
                                 args)
        save_checkpoint(ae,epoch, args,'NNAE','ae')

        ae_loss.append(auto_loss)
        enc_loss.append(encoder_loss)

        print_epoch('NNAE',epoch,time.time()-start,{'AE Loss':auto_loss.numpy(),
                                                    'Encoder Loss':encoder_loss.numpy()},None)

    generate_and_save_training([ae_loss,enc_loss],
                                ['ae loss','encoder loss'],
                                'NNAE',args)
    generate_and_save_images(ae,epoch,image_batch[:25,...],'NNAE',args)

    return ae

def main(train_dataset,train_images,train_labels,test_images,test_labels, test_masks,args):
    ae = Autoencoder(args)
    ae = train(ae,train_dataset, train_images, test_images,test_labels,args)
    end_routine(train_images, test_images, test_labels, test_masks, [ae], 'NNAE', args)

    
if __name__  == '__main__':
    main()
