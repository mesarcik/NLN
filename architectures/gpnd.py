import tensorflow as tf
import numpy as np
import time
from models import (Encoder, 
                    Decoder, 
                    Autoencoder,
                    Discriminator_z,
                    Discriminator_x)
from utils.plotting  import  (generate_and_save_images,
                             generate_and_save_training,
                             save_training_curves)
from utils.training import print_epoch,save_checkpoint
from utils.metrics import get_classifcation,nearest_error,save_metrics
from model_config import BUFFER_SIZE,BATCH_SIZE,cross_entropy

ae_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_x_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_z_optimizer = tf.keras.optimizers.Adam(1e-4)
generator_optimizer = tf.keras.optimizers.Adam(1e-4)

def ae_loss(x,x_hat,loss_weight):
    return loss_weight * cross_entropy(x, x_hat)

def discriminator_loss(real_output, fake_output,loss_weight):
    real_loss =  cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss =  cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return loss_weight * total_loss

def generator_loss(fake_output, loss_weight):
    return  loss_weight * tf.reduce_mean(cross_entropy(tf.ones_like(fake_output), fake_output))

@tf.function
def train_step(ae,discriminator_z,discriminator_x,images,latent_dim):
    noise = tf.random.normal([BATCH_SIZE, latent_dim])

    with tf.GradientTape() as ae_tape, tf.GradientTape() as x_disc_tape,tf.GradientTape() as z_disc_tape,tf.GradientTape() as x_gen_tape, tf.GradientTape() as z_gen_tape:

      x_hat  = ae(images,training=True)  
      z = ae.encoder(images)

      x_real_output,_ = discriminator_x(images, training=True)
      x_fake_output,_ = discriminator_x(x_hat, training=True)

      z_real_output = discriminator_z(noise, training=True)
      z_fake_output = discriminator_z(z, training=True)

      x_disc_loss = discriminator_loss(x_real_output, x_fake_output,1)
      z_disc_loss = discriminator_loss(z_real_output, z_fake_output,1)

      x_gen_loss = generator_loss(x_fake_output,1)
      z_gen_loss = generator_loss(z_fake_output,1)
      auto_loss = ae_loss(images,x_hat,1)


###############################
    gradients_of_discriminator_x = x_disc_tape.gradient(x_disc_loss,
                                                     discriminator_x.trainable_variables)

    gradients_of_generator_x = x_gen_tape.gradient(x_gen_loss, ae.decoder.trainable_variables)

    gradients_of_generator_z = z_gen_tape.gradient(z_gen_loss, ae.encoder.trainable_variables)

    gradients_of_discriminator_z = z_disc_tape.gradient(z_disc_loss,
                                                     discriminator_z.trainable_variables)

    gradients_of_ae= ae_tape.gradient(auto_loss, ae.trainable_variables)
###############################
    discriminator_x_optimizer.apply_gradients(zip(gradients_of_discriminator_x, 
                                                discriminator_x.trainable_variables))

    generator_optimizer.apply_gradients(zip(gradients_of_generator_x, ae.decoder.trainable_variables))
    generator_optimizer.apply_gradients(zip(gradients_of_generator_z, ae.encoder.trainable_variables))
    discriminator_z_optimizer.apply_gradients(zip(gradients_of_discriminator_z, 
                                                discriminator_z.trainable_variables))

    ae_optimizer.apply_gradients(zip(gradients_of_ae, ae.trainable_variables))
    return auto_loss, z_disc_loss,x_disc_loss,x_gen_loss

def train(ae,discriminator_z,discriminator_x,train_dataset,test_images,test_labels,args):
    ae_loss, d_x_loss,d_z_loss ,g_loss,aucs = [],[],[],[],[]
    for epoch in range(args.epochs):
        start = time.time()

        for image_batch in train_dataset:
           auto_loss, disc_x_loss,disc_z_loss, gen_loss =  train_step(ae,
                                                        discriminator_z,
                                                        discriminator_x,
                                                        image_batch,args.latent_dim)

        generate_and_save_images(ae,epoch + 1,image_batch[:25,...],'GPND',args)

        save_checkpoint(ae,epoch,args,'GPND','ae')
        save_checkpoint(discriminator_x,epoch,args,'GPND','disc_x')
        save_checkpoint(discriminator_z,epoch,args,'GPND','disc_z')

        ae_loss.append(auto_loss)
        d_x_loss.append(disc_x_loss)
        d_z_loss.append(disc_z_loss)
        g_loss.append(gen_loss)

        roc_auc,f1 = get_classifcation('GPND',
                                        [ae,discriminator_x],
                                        test_images,
                                        test_labels,
                                        args.anomaly_class,
                                        hera=args.data=='HERA')
        aucs.append(roc_auc)
        print_epoch('GPND',epoch,time.time()-start,{'AE Loss':auto_loss.numpy(),
                                                  'Discriminator x': disc_x_loss.numpy(),
                                                  'Discriminator z': disc_z_loss.numpy()},roc_auc)

    generate_and_save_training([ae_loss,d_x_loss,d_z_loss,g_loss,aucs],
                                ['ae loss',
                                 'discriminator x loss',
                                 'discriminator z loss', 
                                 'generator loss','AUC'],
                                'GPND',args)

    generate_and_save_images(ae,epoch,image_batch[:25,...],'GPND',args)
    return ae, discriminator_x

def main(train_dataset,train_images,train_labels,test_images,test_labels,args):
    ae = Autoencoder(args)
    discriminator_z = Discriminator_z(args)
    discriminator_x = Discriminator_x(args)
    ae, discriminator_x = train(ae,
                                discriminator_z,
                                discriminator_x,
                                train_dataset,
                                test_images,
                                test_labels,
                                args)


    save_training_curves([ae,discriminator_x],
                          args,
                          test_images,
                          test_labels,
                          'GPND')

    auc_latent, f1_latent ,neighbour,radius = nearest_error([ae,discriminator_x],
                                                      train_images,
                                                      test_images,
                                                      test_labels,
                                                      'GPND',
                                                      args,
                                                      args.data == 'HERA')

    auc_recon ,f1_recon = get_classifcation('GPND',
                                             [ae,discriminator_x],
                                             test_images,
                                             test_labels,
                                             args.anomaly_class,
                                             hera = args.data == 'HERA',
                                             f1=True)
    save_metrics('GPND',
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


