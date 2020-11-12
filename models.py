import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from model_config import input_shape,n_layers,n_filters
tf.keras.backend.set_floatx('float32')

class Encoder(tf.keras.layers.Layer):
    def __init__(self,args):
        super(Encoder, self).__init__()
        self.input_layer = layers.InputLayer(input_shape=input_shape)
        self.conv, self.pool, self.batchnorm = [],[],[]
        self.latent_dim  = args.latent_dim

        for n in range(n_layers):
            self.conv.append(layers.Conv2D(filters = n_filters, 
                                       kernel_size = (2,2), 
                                       #strides = (2,2),
                                       padding = 'same',
                                       activation='relu'))
            self.pool.append(layers.MaxPooling2D(pool_size=(2,2),padding='same'))

            self.batchnorm.append(layers.BatchNormalization())

        #output shape = 2,2
        self.flatten = layers.Flatten()
        self.dense_ae = layers.Dense(self.latent_dim, activation=None)

        self.dense_vae = layers.Dense(n_filters, activation='relu')
        self.mean = layers.Dense(self.latent_dim)
        self.logvar = layers.Dense(self.latent_dim)

    def call(self, x,vae=False):
        x = self.input_layer(x)

        for layer in range(n_layers):
            x = self.conv[layer](x)
            if layer !=n_layers-1:
                x = self.pool[layer](x)
            x = self.batchnorm[layer](x)
        x = self.flatten(x)

        if vae: 
            x = self.dense_vae(x)
            mean = self.mean(x)
            logvar = self.logvar(x)
            return [mean,logvar] 
        else: 
            x = self.dense_ae(x)
            return x

class Decoder(tf.keras.layers.Layer):
    def __init__(self,args):
        super(Decoder, self).__init__()
        self.latent_dim = args.latent_dim
        self.input_layer = layers.InputLayer(input_shape=[self.latent_dim,])
        self.dense= layers.Dense(input_shape[0]//2**(n_layers-1) *
                                 input_shape[1]//2**(n_layers-1) *
                                 n_filters,activation='relu')
        self.reshape = layers.Reshape((input_shape[0]//2**(n_layers-1),
                                       input_shape[1]//2**(n_layers-1),
                                       n_filters))

        self.conv, self.pool, self.batchnorm = [],[],[]
        for _ in range(n_layers-1):

            self.conv.append(layers.Conv2DTranspose(filters = n_filters, 
                                               kernel_size = (2,2), 
                                               #strides = (2,2),
                                               padding = 'same',
                                               activation='relu'))

            self.pool.append(layers.UpSampling2D(size=(2,2)))
            self.batchnorm.append(layers.BatchNormalization())

        self.conv_output = layers.Conv2DTranspose(filters = input_shape[-1], 
                                           kernel_size = (2,2), 
                                           padding = 'same',
                                           activation='sigmoid')

    def call(self, x):
        x = self.input_layer(x)
        x = self.dense(x)
        x = self.reshape(x)

        for layer in range(n_layers -1):
            x = self.conv[layer](x)
            x = self.pool[layer](x)
            x = self.batchnorm[layer](x)
        
        x = self.conv_output(x)
        return  x

class Autoencoder(tf.keras.Model):
    def __init__(self,args):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)

    def call(self,x):
        z = self.encoder(x,vae=False)
        x_hat = self.decoder(z)
        return x_hat 

class Discriminator_x(tf.keras.Model):
    def __init__(self,args):
        super(Discriminator_x, self).__init__()
        self.network = Encoder(args)
        self.flatten = layers.Flatten()
        self.dense  = layers.Dense(1,activation='sigmoid')

    def call(self,x):
        z = self.network(x)
        classifier = self.flatten(z) 
        classifier = self.dense(classifier) 
        return z,classifier 

class Discriminator_z(tf.keras.Model):
    def __init__(self,args):
        super(Discriminator_z, self).__init__()
        self.latent_dim = args.latent_dim
        self.model = tf.keras.Sequential()
        self.model.add(layers.InputLayer(input_shape=[self.latent_dim,]))

        self.model.add(layers.Dense(64))
        self.model.add(layers.LeakyReLU())
        self.model.add(layers.Dropout(0.3))

        self.model.add(layers.Dense(128))
        self.model.add(layers.LeakyReLU())
        self.model.add(layers.Dropout(0.3))

        self.model.add(layers.Dense(256))
        self.model.add(layers.LeakyReLU())
        self.model.add(layers.Dropout(0.3))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(1))
    def call(self,x):
        return self.model(x)

class VAE(tf.keras.Model):
    def __init__(self,args):
        super(VAE, self).__init__()
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)
        self.latent_dim = args.latent_dim

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(self.latent_dim))
        return self.decode(eps)#, apply_sigmoid=True)

    def reparameterize(self, mean, logvar):
        batch = tf.shape(mean)[0]
        dim = tf.shape(mean)[1]
        eps = tf.random.normal(shape=(batch, dim))
        return mean + tf.exp(0.5 * logvar) * eps


    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def call(self,x):
        mean, logvar = self.encoder(x,vae=True)
        z = self.reparameterize(mean, logvar)
        x_hat = self.decode(z)
        return x_hat 

class Remap(tf.keras.Model):
    def __init__(self,args):
        super(Remap, self).__init__()
        self.latent_dim = args.latent_dim
        self.input_layer = layers.InputLayer(input_shape=[self.latent_dim,])
        self.dense_layers = []

        for layer in range(3):
            self.dense_layers.append(layers.Dense(32*(layer+1),activation='relu'))

        self.flatten = layers.Flatten()
        self.dense = layers.Dense(self.latent_dim, activation=None)

    def call(self, x):
        x = self.input_layer(x)

        for layer in range(3):
            x = self.dense_layers[layer](x)
        
        x = self.flatten(x)
        x = self.dense(x)
        return  x

class Discriminator_bigan(tf.keras.Model):
    def __init__(self):
        super(Discriminator_bigan, self).__init__()
        self.network = Encoder()
        self.flatten = layers.Flatten()

        self.dense_z= layers.Dense(n_filters*8,activation='relu')
        self.flatten_z= layers.Flatten()

        self.dense_concat = layers.Dense(1024, activation='relu')
        self.output_concat = layers.Dense(1,activation='sigmoid')

    def call(self,x,z):
        o_x = self.network(x,vae=False)
        o_z = self.dense_z(z)
        o_z = self.flatten_z(o_z)
        o = layers.concatenate([o_x,o_z])
        o = self.dense_concat(o) 

        return self.output_concat(o),o 

