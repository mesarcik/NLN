#config for training
import tensorflow as tf

BUFFER_SIZE =25000 # 60000
BATCH_SIZE = 2**10
cross_entropy = tf.keras.losses.BinaryCrossentropy()
mse = tf.keras.losses.MeanSquaredError()

### AE Parameters for CIFAR, MNIST, FMNIST 
n_filters =32 
n_layers =3
