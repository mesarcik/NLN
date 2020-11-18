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

    return model 

def get_error(name,model,test_images,return_z=False):
    if name == 'AE':
        model_output = model(test_images)
        z = model.encoder(test_images)
        error = np.abs(test_images - model_output.numpy())

    elif name == 'AAE':
        model_output = model(test_images)
        z= model.encoder(test_images)
        error = np.abs(test_images- model_output.numpy())

    elif name == 'DAE':
        model_output = model[0](test_images)
        z = model[0].encoder(test_images)
        error = np.abs(test_images -  model_output.numpy())

    elif name  == 'DAE_disc':
        x_hat = model[0](test_images)
        z = model[0].encoder(test_images)

        d_x, _ = model[1](test_images)
        d_x_hat, _ = model[1](x_hat)
        
        reconstruction_error = np.abs(test_images, x_hat.numpy())
        reconstruction_error = reconstruction_error.mean(
                                                         axis=tuple(
                                                         range(1,reconstruction_error.ndim)))

        discriminator_error  = np.abs(d_x.numpy() - d_x_hat.numpy())
        discriminator_error = discriminator_error.mean(
                                                         axis=tuple(
                                                         range(1,discriminator_error.ndim)))
        alpha = 0.9
        error = (1-alpha)*reconstruction_error + alpha*discriminator_error

    elif name == 'VAE':
        z_mean, z_log_var = model.encoder(test_images,vae=True)
        z = model.reparameterize(z_mean, z_log_var)
        model_output = model.decoder(z)
        error = np.abs(test_images- model_output.numpy())

    elif name == 'GANomaly':
        x_hat = model[0](test_images)

        z = model[0].encoder(test_images)
        z_hat = model[2](x_hat)
        
        error = np.abs(z.numpy()- z_hat.numpy())

    elif name  == 'GANomaly_disc':
        x_hat = model[0](test_images)

        z = model[0].encoder(test_images)
        z_hat = model[2](x_hat)

        d_x, _ = model[1](test_images)
        d_x_hat, _ = model[1](x_hat)
        
        reconstruction_error = np.mean(np.abs(z.numpy() - z_hat.numpy()))
        discriminator_error = np.mean(np.abs(d_x.numpy() - d_x_hat.numpy()))
        
        alpha = 0.9
        error = (1-alpha)*reconstruction_error + alpha*discriminator_error

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
        z_mean, z_logvar  = model.encoder(test_images,vae=True)
        z = model.reparameterize(z_mean, z_logvar)
        model_output = model.decoder(z)

    elif name == 'GANomaly' or name == 'GANomaly_disc':
        model_output = model[0](test_images)

    return model_output 
