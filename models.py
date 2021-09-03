import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from model_config import n_layers,n_filters
tf.keras.backend.set_floatx('float32')

class Encoder(tf.keras.layers.Layer):
    def __init__(self,args):
        super(Encoder, self).__init__()
        self.input_layer = layers.InputLayer(input_shape=args.input_shape)
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
        self.dense= layers.Dense(args.input_shape[0]//2**(n_layers-1) *
                                 args.input_shape[1]//2**(n_layers-1) *
                                 n_filters,activation='relu')
        self.reshape = layers.Reshape((args.input_shape[0]//2**(n_layers-1),
                                       args.input_shape[1]//2**(n_layers-1),
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

        self.conv_output = layers.Conv2DTranspose(filters = args.input_shape[-1], 
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
class MultiEncoder(tf.keras.Model):
    def __init__(self,args):
        super(MultiEncoder, self).__init__()
        self.input_layer = layers.InputLayer(input_shape=args.input_shape)
        self.latent_dim  = args.latent_dim
        self.nneighbours = args.neighbors[0]
        self.input_convs  = []
        for n in range(self.nneighbours+1):
            self.input_convs.append([layers.Conv2D(n_filters, (4, 4), strides=2, activation=layers.LeakyReLU(alpha=0.2), padding='same'), 
                                     layers.Conv2D(n_filters, (4, 4), strides=2, activation=layers.LeakyReLU(alpha=0.2), padding='same')])

        self.concat = []

        self.conv = []
        self.conv.append(layers.Conv2D(n_filters, (4, 4), strides=2, activation=layers.LeakyReLU(alpha=0.2), padding='same'))
        self.conv.append(layers.Conv2D(n_filters, (3, 3), strides=1, activation=layers.LeakyReLU(alpha=0.2), padding='same'))
        self.conv.append(layers.Conv2D(n_filters*2, (4, 4), strides=2, activation=layers.LeakyReLU(alpha=0.2), padding='same'))
        self.conv.append(layers.Conv2D(n_filters*2, (3, 3), strides=1, activation=layers.LeakyReLU(alpha=0.2), padding='same'))
        self.conv.append(layers.Conv2D(n_filters*4, (4, 4), strides=2, activation=layers.LeakyReLU(alpha=0.2), padding='same'))
        self.conv.append(layers.Conv2D(n_filters*2, (3, 3), strides=1, activation=layers.LeakyReLU(alpha=0.2), padding='same'))
        self.conv.append(layers.Conv2D(n_filters, (3, 3), strides=1, activation=layers.LeakyReLU(alpha=0.2), padding='same'))
        self.conv.append(layers.Conv2D(self.latent_dim, (8, 8), strides=1, activation='linear', padding='valid'))

        self.reshape = layers.Reshape((self.latent_dim,))

    def call(self, x, nln):
        """
            x (np.array): train batch
            nln (tuple/list): a list of inputs to MultiEncoder in the following order [ NLN_0,...,NLN_n]
        """
        outputs = []
        x = self.input_convs[0][0](x)
        outputs.append(self.input_convs[0][1](x))

        for n, inp in enumerate(nln):
            x = self.input_convs[n+1][0](inp)
            outputs.append(self.input_convs[n+1][1](x))
        x = layers.concatenate(outputs)

        for layer in self.conv:
            x = layer(x)
        x = self.reshape(x)

        return x
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

class VectorQuantizer(layers.Layer):
    # Note taken from https://keras.io/examples/generative/vq_vae/
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = (
            beta  # This parameter is best kept between [0.25, 2] as per the paper.
        )

        # Initialize the embeddings which we will quantize.
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape=(self.embedding_dim, self.num_embeddings), dtype="float32"
            ),
            trainable=True,
            name="embeddings_vqvae",
        )

    def call(self, x):
        # Calculate the input shape of the inputs and
        # then flatten the inputs keeping `embedding_dim` intact.
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dim])

        # Quantization.
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
        quantized = tf.reshape(quantized, input_shape)

        # Calculate vector quantization loss and add that to the layer. You can learn more
        # about adding losses to different layers here:
        # https://keras.io/guides/making_new_layers_and_models_via_subclassing/. Check
        # the original paper to get a handle on the formulation of the loss function.
        commitment_loss = self.beta * tf.reduce_mean(
            (tf.stop_gradient(quantized) - x) ** 2
        )
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(commitment_loss + codebook_loss)

        # Straight-through estimator.
        quantized = x + tf.stop_gradient(quantized - x)
        return quantized

    def get_code_indices(self, flattened_inputs):
        # Calculate L2-normalized distance between the inputs and the codes.
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
            tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings ** 2, axis=0)
            - 2 * similarity
        )

        # Derive the indices for minimum distances.
        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices


class VQVAE(tf.keras.Model):
    def __init__(self,args):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)
        self.latent_dim = args.latent_dim
        self.vq = VectorQuantizer(64, args.latent_dim, name="vector_quantizer")

    def call(self,x):
        z = self.encoder(x,vae=False)
        z_q = self.vq(z)
        x_hat = self.decoder(z_q)
        return x_hat 
