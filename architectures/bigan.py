import tensorflow as tf
import numpy as np
import os
import time
from models import (Encoder, 
                    Decoder, 
                    Discriminator_bigan)
from utils.plotting  import  (generate_and_save_images,
                             generate_and_save_training,
                             save_training_curves)
from utils.training import print_epoch,save_checkpoint
from utils.metrics import get_classifcation,nearest_error,save_metrics
from model_config import BUFFER_SIZE,BATCH_SIZE,cross_entropy


discriminator_optimizer = tf.keras.optimizers.Adam(1e-5,beta_1=0.5)
generator_optimizer = tf.keras.optimizers.Adam(1e-5,beta_1=0.5)


def discriminator_loss(real_output, fake_output,loss_weight):
    real_loss =  cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss =  cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return loss_weight * total_loss

def generator_loss(real_output,fake_output,loss_weight):
    real_loss =  cross_entropy(tf.zeros_like(real_output), real_output)
    fake_loss =  cross_entropy(tf.ones_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return loss_weight * total_loss

@tf.function
def train_step(encoder,decoder,discriminator,images):

    with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:

      z_fake = encoder(images,training=True)
      z_real =  tf.random.normal(z_fake.shape)

      x_fake = decoder(z_real,training=True)
      x_real =  images

      real_output,_ = discriminator(x_real,z_fake, training=True) #encoder loss
      fake_output,_ = discriminator(x_fake,z_real, training=True) #decoder loss

      disc_loss = discriminator_loss(real_output, fake_output,1)
      gen_loss = generator_loss(real_output,fake_output,1)

    gradients_of_gen = gen_tape.gradient(gen_loss, encoder.trainable_variables + decoder.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_gen, encoder.trainable_variables
                                                            + decoder.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return  gen_loss,disc_loss,decoder(encoder(images[:25,...],training=False),training=False)

def train(encoder,decoder,discriminator,train_dataset,test_images,test_labels,args):
    generator_loss,discriminator_loss,aucs =  [],[],[]

    for epoch in range(args.epochs):
        start = time.time()

        for image_batch in train_dataset:
           gen_loss,disc_loss,o =  train_step(encoder,decoder,discriminator,image_batch)

        generate_and_save_images(None,epoch + 1,o,'BIGAN',args)
        save_checkpoint(encoder,epoch,args,'BIGAN','enc')
        save_checkpoint(decoder,epoch,args,'BIGAN','dec')
        save_checkpoint(discriminator,epoch,args,'BIGAN','disc')

        generator_loss.append(gen_loss)
        discriminator_loss.append(disc_loss)

        roc_auc,f1 = get_classifcation('BIGAN',
                                        [encoder,decoder,discriminator],
                                        test_images,
                                        test_labels,
                                        args.anomaly_class,
                                        hera=args.data =='HERA')
        aucs.append(roc_auc)
        print_epoch('BIGAN',
                    epoch,
                    time.time()-start,
                    {'Discriminator loss': disc_loss.numpy(),
                     'Generator loss':gen_loss.numpy()},
                    roc_auc)


    generate_and_save_training([discriminator_loss,generator_loss],
                                ['discriminator loss','generator loss'],
                                'BIGAN',args)
    generate_and_save_images(None,epoch, o,'BIGAN',args)
    return encoder,decoder, discriminator
    
def main(train_dataset,train_images,train_labels,test_images,test_labels,args):

    decoder = tf.keras.Sequential(Decoder(args))
    encoder = tf.keras.Sequential(Encoder(args))
    discriminator = Discriminator_bigan()

    encoder, decoder, discriminator = train(encoder,
                                            decoder,
                                            discriminator,
                                            train_dataset,
                                            test_images,
                                            test_labels,
                                            args)

    save_training_curves([encoder,decoder,discriminator],
                         args,
                         test_images,
                         test_labels,
                         'BIGAN')

    auc_latent, f1_latent,neighbour,radius  = nearest_error([encoder,decoder,discriminator],
                                                      train_images,
                                                      test_images,
                                                      test_labels,
                                                      'BIGAN',
                                                      args,
                                                      args.data == 'HERA')

    auc_recon ,f1_recon = get_classifcation('BIGAN',
                                            [encoder,decoder,discriminator],
                                             test_images,
                                             test_labels,
                                             args.anomaly_class,
                                             hera = args.data == 'HERA',
                                             f1=True)
    save_metrics('BIGAN',
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


