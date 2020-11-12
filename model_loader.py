from models import *
import tensorflow as tf

def get_model(name = 'AE',anomaly = None):
    if name == 'AE':
        model = Autoencoder()
        model.load_weights('outputs/AE/{}/training_checkpoints/checkpoint_ae'.format(anomaly))
    elif name == 'AAE':
        model = Autoencoder()
        model.load_weights('outputs/AAE/{}/training_checkpoints/checkpoint_ae'.format(anomaly))
    elif name == 'VAE':
        model = VAE()
        model.load_weights('outputs/VAE/{}/training_checkpoints/checkpoint_ae'.format(anomaly))

    elif name == 'DAE' or name == 'DAE_disc':
        ae = Autoencoder()
        disc= Discriminator_x()
        ae.load_weights('outputs/DAE/{}/training_checkpoints/checkpoint_ae'.format(anomaly))
        disc.load_weights('outputs/DAE/{}/training_checkpoints/checkpoint_disc'.format(anomaly))
        model = [ae,disc]

    elif name == 'GANomaly' or name == 'GANomaly_disc':
        ae = Autoencoder()
        disc= Discriminator_x()
        encoder = tf.keras.Sequential(Encoder())
        ae.load_weights('outputs/GANomaly/{}/training_checkpoints/checkpoint_ae'.format(
                                                                                    anomaly))
        disc.load_weights('outputs/GANomaly/{}/training_checkpoints/checkpoint_disc'.format(
                                                                                       anomaly))
        encoder.load_weights('outputs/GANomaly/{}/training_checkpoints/checkpoint_encoder'.format(anomaly))

        model = [ae,disc,encoder]
    elif name == 'BIGAN':
        enc= tf.keras.Sequential(Encoder())
        dec= tf.keras.Sequential(Decoder())
        disc = Discriminator_bigan()
        enc.load_weights('outputs/BIGAN/{}/training_checkpoints/checkpoint_enc'.format(anomaly))
        dec.load_weights('outputs/BIGAN/{}/training_checkpoints/checkpoint_dec'.format(anomaly))
        disc.load_weights('outputs/BIGAN/{}/training_checkpoints/checkpoint_disc'.format(anomaly))
        model = [enc,dec,disc]
    elif name == 'GPND':
        ae = Autoencoder()
        disc_x= Discriminator_x()
        ae.load_weights('outputs/GPND/{}/training_checkpoints/checkpoint_ae'.format(anomaly))
        disc_x.load_weights('outputs/GPND/{}/training_checkpoints/checkpoint_disc_x'.format(
                                                                                        anomaly))
        model = [ae,disc_x]
        
    elif name == 'VAEGAN':
        vae = VAE()
        disc = Discriminator_x()
        vae.load_weights('outputs/VAEGAN/{}/training_checkpoints/checkpoint_vae'.format(anomaly))
        disc.load_weights('outputs/VAEGAN/{}/training_checkpoints/checkpoint_disc'.format(anomaly))
        model = [vae,disc]

    return model 

def get_error(name,model,test_images,return_z=False):
    if name == 'AE':
        model_output = model(test_images)
        error = tf.square(tf.subtract(test_images, model_output)).numpy()
        z = model.encoder(test_images)

    elif name == 'AAE':
        model_output = model(test_images)
        error = tf.square(tf.subtract(test_images, model_output)).numpy()
        z= model.encoder(test_images)

    elif name == 'DAE':
        model_output = model[0](test_images)
        error = tf.square(tf.subtract(test_images, model_output)).numpy()
        z = model[0].encoder(test_images)

    elif name  == 'DAE_disc':
        x_hat = model[0](test_images)
        z = model[0].encoder(test_images)

        d_x, _ = model[1](test_images)
        d_x_hat, _ = model[1](x_hat)
        
        reconstruction_error = tf.math.abs(tf.subtract(test_images, x_hat)).numpy()
        reconstruction_error = reconstruction_error.mean(
                                                         axis=tuple(
                                                         range(1,reconstruction_error.ndim)))

        discriminator_error  = tf.square(tf.subtract(d_x, d_x_hat)).numpy()
        discriminator_error = discriminator_error.mean(
                                                         axis=tuple(
                                                         range(1,discriminator_error.ndim)))
        alpha = 0.9
        error = (1-alpha)*reconstruction_error + alpha*discriminator_error

    elif name == 'VAE':
        z_mean, z_log_var = model.encoder(test_images,vae=True)
        z = model.reparameterize(z_mean, z_log_var)
        model_output = model.decoder(z)
        error = tf.square(tf.subtract(test_images, model_output)).numpy()

       # cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
       # N = 2# number of samples - determine which number to use (?)
       # z_mean, z_log_var  = model.encoder(test_images,vae=True)

       # error = np.empty(test_images.shape[0])
       # for i,image in enumerate(test_images):
       #     temp = np.empty([N])
       #     z = np.array([np.random.multivariate_normal(mean = z_mean[i,...],
       #                                   cov = np.diag(np.exp(z_log_var[i,...])))
       #                                   for i in range(N)])
       #     imgs = np.repeat(np.expand_dims(image,axis=0),N,axis=0)

       #     error[i] =cross_entropy(image,model.decoder(z))


    elif name == 'GANomaly':
        x_hat = model[0](test_images)

        z = model[0].encoder(test_images)
        z_hat = model[2](x_hat)
        
        error = tf.math.abs(tf.subtract(z, z_hat)).numpy()

    elif name  == 'GANomaly_disc':
        x_hat = model[0](test_images)

        z = model[0].encoder(test_images)
        z_hat = model[2](x_hat)

        d_x, _ = model[1](test_images)
        d_x_hat, _ = model[1](x_hat)
        
        reconstruction_error = np.mean(tf.math.abs(tf.subtract(z, z_hat)).numpy())
        discriminator_error = np.mean(tf.square(tf.subtract(d_x, d_x_hat)).numpy())
        
        alpha = 0.9
        error = (1-alpha)*reconstruction_error + alpha*discriminator_error

    elif name == 'BIGAN':
        z_hat = model[0](test_images)
        z_real =  tf.random.normal(z_hat.shape)
        x_hat = model[1](z_hat)
        _,d_z = model[2](test_images,z_hat)
        _,d_z_hat = model[2](x_hat,z_hat)

        reconstruction_error = tf.square(tf.subtract(test_images, x_hat)).numpy()
        discriminator_error = tf.square(tf.subtract(d_z, d_z_hat)).numpy()
        error = ((1-0.9)*reconstruction_error.mean(axis=tuple(range(1,reconstruction_error.ndim))) # TODO: alpha was 0.7
                 + 0.9*discriminator_error.mean(axis=tuple(range(1,discriminator_error.ndim))))

        z = model[0](test_images)


    elif name == 'GPND':
        x_hat = model[0](test_images)
        d_x_hat,_ = model[1](x_hat)
        d_x,_ = model[1](test_images)

        reconstruction_error = tf.square(tf.subtract(test_images, x_hat)).numpy()
        discriminator_error = tf.square(tf.subtract(d_x, d_x_hat)).numpy()
        error = ((1-0.7)*reconstruction_error.mean(axis=tuple(range(1,reconstruction_error.ndim)))
              + 0.7*discriminator_error.mean(axis=tuple(range(1,discriminator_error.ndim))))
        z = model[0].encoder(test_images)

    elif name == 'VAEGAN':
        x_hat = model[0](test_images)
        d_x_hat,_ = model[1](x_hat)
        d_x,_ = model[1](test_images)

        reconstruction_error = tf.square(tf.subtract(test_images, x_hat)).numpy()
        discriminator_error = tf.square(tf.subtract(d_x, d_x_hat)).numpy()
        error = ((1-0.7)*reconstruction_error.mean(axis=tuple(range(1,reconstruction_error.ndim)))
                  + 0.7*discriminator_error.mean(axis=tuple(range(1,discriminator_error.ndim))))

        z = model[0].encoder(test_images)

    error =  error.mean(axis=tuple(range(1,error.ndim)))
    if return_z:
        return z, error
    else:
        return error 

def get_reconstructed(name,model,test_images):
    if name == 'AE':
        model_output = model(test_images)

    elif name == 'AAE':
        model_output = model(test_images)

    elif name == 'DAE' or name == 'DAE_disc':
        model_output = model[0](test_images)

    elif name == 'VAE':
        N = 2# number of samples - determine which number to use (?)
        z_mean, z_logvar  = model.encoder(test_images,vae=True)
        z = model.reparameterize(z_mean, z_logvar)
        model_output = model.decoder(z)

    elif name == 'GANomaly' or name == 'GANomaly_disc':
        model_output = model[0](test_images)

    elif name == 'BIGAN':
        z_hat = model[0](test_images)
        model_output = model[1](z_hat)

    elif name == 'GPND':
         model_output = model[0](test_images)

    elif name == 'VAEGAN':
        model_output  = model[0](test_images)

    return model_output 
