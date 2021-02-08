#config for training
import tensorflow as tf

BUFFER_SIZE =25000 # 60000
BATCH_SIZE = 256 #128
cross_entropy = tf.keras.losses.BinaryCrossentropy()

n_layers = 2
n_filters = 32
