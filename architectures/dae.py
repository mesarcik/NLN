import tensorflow as tf
import numpy as np
import time
from models import (Encoder, 
                   Decoder, 
                   Autoencoder,
                   Discriminator_x)

from utils.plotting  import  (generate_and_save_images,
                             generate_and_save_training,
                             save_training_curves)

from utils.training import print_epoch,save_checkpoint
from utils.metrics import get_classifcation,nearest_error,save_metrics
from model_config import *

ae_optimizer = tf.keras.optimizers.Adam()#1e-5)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-5)
generator_optimizer = tf.keras.optimizers.Adam(1e-5)

def ae_loss(x,x_hat):
    return cross_entropy(x,x_hat)
    #return tf.reduce_mean(tf.square(tf.subtract(x, x_hat)))

def discriminator_loss(real_output, fake_output,loss_weight):
    real_loss =  cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss =  cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return loss_weight * total_loss

def generator_loss(fake_output, loss_weight):
    return  loss_weight * tf.reduce_mean(cross_entropy(tf.ones_like(fake_output), fake_output))

@tf.function
def train_step(ae,discriminator, x):
    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape() as ae_tape,\
         tf.GradientTape() as disc_tape,\
         tf.GradientTape() as gen_tape:

        x_hat = ae(x)

        real_output,c0 = discriminator(x, training=True)
        fake_output,c1 = discriminator(x_hat, training=True)

        auto_loss = ae_loss(x,x_hat)
        disc_loss = discriminator_loss(real_output, fake_output,1)
        gen_loss = generator_loss(fake_output,1)

    gradients_of_ae = ae_tape.gradient(auto_loss, ae.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, 
                                                discriminator.trainable_variables)
    gradients_of_generator = gen_tape.gradient(gen_loss, 
                                                ae.decoder.trainable_variables)

    ae_optimizer.apply_gradients(zip(gradients_of_ae, ae.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, 
                                                discriminator.trainable_variables))

    generator_optimizer.apply_gradients(zip(gradients_of_generator, 
                                            ae.decoder.trainable_variables))
    return auto_loss,disc_loss,gen_loss

def train(ae,discriminator, train_dataset,test_images,test_labels,args):
    ae_loss,d_loss, g_loss,aucs = [], [], [],[]
    for epoch in range(args.epochs):
        start = time.time()
        for image_batch in train_dataset:
           auto_loss,disc_loss,gen_loss  =  train_step(ae, 
                                                       discriminator, 
                                                       image_batch)

        generate_and_save_images(ae,
                                 epoch + 1,
                                 image_batch[:25,...],
                                 'DAE_disc',
                                 args)

        save_checkpoint(ae,epoch,args,'DAE_disc','ae')
        save_checkpoint(discriminator, epoch, args,'DAE_disc','disc')

        ae_loss.append(auto_loss)
        d_loss.append(disc_loss)
        g_loss.append(gen_loss)

        roc_auc,f1 = get_classifcation('DAE_disc',
                                       [ae,discriminator],
                                       test_images,
                                       test_labels,
                                       args.anomaly_class,
                                       hera=args.data=='HERA')
        aucs.append(roc_auc)

        print_epoch('DAE_disc',
                     epoch,
                     time.time()-start,
                     {'AE Loss':auto_loss.numpy(),
                      'Discrimator Loss':disc_loss.numpy(),
                      'Generator Loss':gen_loss.numpy()},
                     roc_auc)

    generate_and_save_training([ae_loss,d_loss,g_loss,aucs],
                                ['ae loss','disc loss','gen loss','AUC'],
                                'DAE_disc',args)

    generate_and_save_images(ae,epoch,image_batch[:25,...],'DAE_disc',args)

    return ae,discriminator

def main(train_dataset,train_images,train_labels,test_images,test_labels,args):
    ae = Autoencoder(args)
    discriminator = Discriminator_x(args)
    ae, discriminator = train(ae,
                              discriminator,
                              train_dataset,
                              test_images,
                              test_labels,
                              args)

    save_training_curves([ae,discriminator],
                         args,
                         test_images,
                         test_labels,
                         'DAE_disc')

    auc_latent, f1_latent, neighbour,radius = nearest_error([ae,discriminator],
                                                             train_images,
                                                             test_images,
                                                             test_labels,
                                                             'DAE_disc',
                                                             args,
                                                             args.data == 'HERA')
    
    auc_recon ,f1_recon = get_classifcation('DAE_disc',
                                             [ae,discriminator],
                                             test_images,
                                             test_labels,
                                             args.anomaly_class,
                                             hera = args.data == 'HERA',
                                             f1=True)
    save_metrics('DAE_disc',
                 args.model_name,
                 args.anomaly_class,
                 auc_recon, 
                 f1_recon,
                 neighbour,
                 radius,
                 auc_latent,
                 f1_latent)
    
if __name__  == '__main__':
    main()
