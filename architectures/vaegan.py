import tensorflow as tf
import numpy as np
import time
from models import (Encoder, 
                    Decoder, 
                    VAE, 
                    Discriminator_x)
from utils.plotting  import  (generate_and_save_images,
                             generate_and_save_training,
                             save_training_curves)
from utils.training import print_epoch,save_checkpoint
from utils.metrics import get_classifcation,nearest_error,save_metrics
from model_config import BUFFER_SIZE,BATCH_SIZE,cross_entropy

kl = tf.keras.losses.KLDivergence()
encoder_optimizer = tf.keras.optimizers.Adam()
decoder_optimizer = tf.keras.optimizers.Adam(1e-5)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-5)

def KL_loss(logvar,mean):
     kl_loss = 1 + logvar - tf.square(mean) - tf.exp(logvar)
     kl_loss = -0.5*tf.reduce_sum(kl_loss,axis=-1)
     return kl_loss

def compute_loss(model, x):
    mean, logvar = model.encoder(x,vae=True)
    z = model.reparameterize(mean, logvar)
    x_hat = model.decode(z)
    var_loss = KL_loss(logvar,mean)
    reconstruction_loss =  ae_loss(x,x_hat,32*32)
    return z,x_hat,var_loss,tf.keras.backend.mean(var_loss + reconstruction_loss)

def gan_loss(real_output,fake_output, loss_weight):
    real_loss = tf.compat.v1.losses.sigmoid_cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = tf.compat.v1.losses.sigmoid_cross_entropy(tf.zeros_like(fake_output), fake_output)

    return  loss_weight * (real_loss + fake_loss )

def ae_loss(x,x_hat,loss_weight):
     return loss_weight*cross_entropy(tf.keras.backend.flatten(x),tf.keras.backend.flatten(x_hat))

def generator_loss(fake_output, loss_weight):
    return  loss_weight * tf.reduce_mean(cross_entropy(tf.ones_like(fake_output), fake_output))

@tf.function
def train_step(vae,discriminator,images,latent_dim):
    noise = tf.random.normal([BATCH_SIZE, latent_dim])

    with tf.GradientTape() as encoder_tape, tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
        
      z,x_hat, var_loss,encoder_loss = compute_loss(vae,images)
      x_p = vae.decoder(noise)

      real_output,real_output_p = discriminator(images, training=True)
      fake_output,fake_output_p = discriminator(x_hat, training=True)
      _,x_output_p = discriminator(x_p, training=True)

      g_loss = gan_loss(real_output_p,fake_output_p,1)
      decoder_loss = generator_loss(fake_output_p,1)



    gradients_of_encoder = encoder_tape.gradient(encoder_loss, vae.trainable_variables)
    gradients_of_generator = gen_tape.gradient(decoder_loss, vae.decoder.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(g_loss, discriminator.trainable_variables)

    encoder_optimizer.apply_gradients(zip(gradients_of_encoder, vae.trainable_variables))
    decoder_optimizer.apply_gradients(zip(gradients_of_generator, vae.decoder.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return encoder_loss, decoder_loss ,g_loss

def train(vae,discriminator,train_dataset,test_images,test_labels,args):
    vae_loss, d_loss,g_loss,aucs = [],[],[],[]
    for epoch in range(args.epochs):
        start = time.time()

        for image_batch in train_dataset:
           var_loss, disc_loss, gan_loss =  train_step(vae,discriminator,image_batch,args.latent_dim)

        generate_and_save_images(vae,epoch + 1,image_batch[:25,...],'VAEGAN',args)
        save_checkpoint(vae,epoch,args,'VAEGAN','vae')
        save_checkpoint(discriminator,epoch,args,'VAEGAN','disc')

        vae_loss.append(var_loss)
        d_loss.append(disc_loss)
        g_loss.append(gan_loss)

        roc_auc,f1 = get_classifcation('VAEGAN',
                                        [vae,discriminator],
                                        test_images,
                                        test_labels,
                                        args.anomaly_class,
                                        hera=args.data=='HERA')
        aucs.append(roc_auc)
        print_epoch('VAEGAN',
                     epoch,
                     time.time()-start,
                     {'VAE Loss':var_loss.numpy(),
                      'Discriminator loss': disc_loss.numpy(),
                      'Generator loss':gan_loss.numpy()},
                     roc_auc)

    generate_and_save_training([vae_loss,d_loss,g_loss,aucs],
                                ['Encoder Loss', 'Decoder Loss', 'Discrimiator loss','AUC'],
                                'VAEGAN',args)
    generate_and_save_images(vae,epoch,image_batch[:25,...],'VAEGAN',args)
    return vae,discriminator

def main(train_dataset,train_images,train_labels,test_images,test_labels,args):
    vae = VAE(args)
    discriminator = Discriminator_x(args)

    vae,discriminator = train(vae,
                              discriminator,
                              train_dataset,
                              test_images,
                              test_labels,
                              args)


    save_training_curves([vae,discriminator],
                          args,
                          test_images,
                          test_labels,
                          'VAEGAN')

    auc_latent, f1_latent, neighbour,radius = nearest_error([vae,discriminator],
                                                       train_images,
                                                       test_images,
                                                       test_labels,
                                                       'VAEGAN',
                                                       args,
                                                       args.data == 'HERA')

    auc_recon ,f1_recon = get_classifcation('VAEGAN',
                                             [vae,discriminator],
                                             test_images,
                                             test_labels,
                                             args.anomaly_class,
                                             hera = args.data == 'HERA',
                                             f1=True)
    save_metrics('VAEGAN',
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


