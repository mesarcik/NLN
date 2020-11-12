#config for training
import tensorflow as tf

BUFFER_SIZE =25000 # 60000
BATCH_SIZE = 128#256
cross_entropy = tf.keras.losses.BinaryCrossentropy()

input_shape = (28,28,1) #TODO automatically set this based on training set 
n_layers = 3
n_filters = 32
