from models import *
import tensorflow as tf

def get_error(name,model,test_images,return_z=False):
    """
        Gets the reconstruciton error of a given model 

        name (str) model name
        model (tf.keras.Model) the model used to compute the reconstrucion error
        test_images (np.array) the test images to compute the error with
        return_z (boolean) return the latent vector 
    """

    if name == 'AE':
        model_output = model(test_images)
        z = model.encoder(test_images)
        error = (test_images - model_output.numpy())

    elif name == 'AAE':
        model_output = model(test_images)
        z= model.encoder(test_images)
        error = (test_images- model_output.numpy())

    elif name == 'DAE':
        model_output = model[0](test_images)
        z = model[0].encoder(test_images)
        error = (test_images -  model_output.numpy())

    elif name  == 'DAE_disc':
        x_hat = model[0](test_images)
        z = model[0].encoder(test_images)

        d_x, _ = model[1](test_images)
        d_x_hat, _ = model[1](x_hat)
        
        reconstruction_error = (test_images, x_hat.numpy())
        reconstruction_error = reconstruction_error.mean(
                                                         axis=tuple(
                                                         range(1,reconstruction_error.ndim)))

        discriminator_error  = (d_x.numpy() - d_x_hat.numpy())
        discriminator_error = discriminator_error.mean(
                                                         axis=tuple(
                                                         range(1,discriminator_error.ndim)))
        alpha = 0.9
        error = (1-alpha)*reconstruction_error + alpha*discriminator_error

    elif name == 'VAE':
        z_mean, z_log_var = model.encoder(test_images,vae=True)
        z = model.reparameterize(z_mean, z_log_var)
        model_output = model.decoder(z)
        error = (test_images- model_output.numpy())

    elif name == 'GANomaly':
        x_hat = model[0](test_images)

        z = model[0].encoder(test_images)
        z_hat = model[2](x_hat)
        
        error = (z.numpy()- z_hat.numpy())

    elif name  == 'GANomaly_disc':
        x_hat = model[0](test_images)

        z = model[0].encoder(test_images)
        z_hat = model[2](x_hat)

        d_x, _ = model[1](test_images)
        d_x_hat, _ = model[1](x_hat)
        
        reconstruction_error = np.mean((z.numpy() - z_hat.numpy()))
        discriminator_error = np.mean((d_x.numpy() - d_x_hat.numpy()))
        
        alpha = 0.9
        error = (1-alpha)*reconstruction_error + alpha*discriminator_error

    error =  error.mean(axis=tuple(range(1,error.ndim)))
    if return_z:
        return z, error
    else:
        return error 

def get_reconstructed(name,model,test_images):
    """
       Gets reconstructions of a model 

        name (str) model name
        model (tf.keras.Model) the model used to compute the reconstrucion error
        test_images (np.array) the test images to compute the error with
    """

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
