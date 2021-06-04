#config for training
import tensorflow as tf

BUFFER_SIZE =25000 # 60000
BATCH_SIZE = 256 
cross_entropy = tf.keras.losses.BinaryCrossentropy()
mse = tf.keras.losses.MeanSquaredError()

# TODO: need to update this to change with input dimensions
resnet = tf.keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(256,256,3),
        pooling='max',)


n_filters = 32
n_layers =3
