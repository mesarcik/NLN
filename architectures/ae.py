import tensorflow as tf
import numpy as np
from sklearn import neighbors
from matplotlib import pyplot as plt
import time
from models import (Encoder, 
                   Decoder, 
                   Autoencoder)

from utils.plotting  import  (generate_and_save_images,
                             generate_and_save_training)

from utils.training import print_epoch,save_checkpoint
from model_config import *
from .helper import end_routine
from inference import infer

optimizer = tf.keras.optimizers.Adam(2e-4)
NNEIGHBOURS= 5

def l2_loss(x,x_hat):

    return mse(x,x_hat)


@tf.function
def train_step(model, x):
    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape() as tape:
        x_hat = model(x)
        loss = l2_loss(x,x_hat)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def train(ae,train_dataset,train_images, test_images,test_labels,args,verbose=True,save=True):
    ae_loss= []
    for epoch in range(args.epochs):
        start = time.time()

        #_z = infer(ae.encoder, train_images, args, 'encoder')
        #_x = infer(ae , train_images, args, 'AE')
        #nbrs = neighbors.NearestNeighbors(n_neighbors= NNEIGHBOURS, algorithm='ball_tree', n_jobs=-1).fit(_z) 

        #neighbours_dist, neighbours_idx  =  nbrs.kneighbors(_z, return_distance=True)
        #neighbours = _x[neighbours_idx]

        ############ DELETE 
        #fig,axs = plt.subplots(5,NNEIGHBOURS +2,figsize=(5,5))
        #for i in range(5):
        #    r = np.random.randint(train_images.shape[0])
        #    axs[i,0].imshow(train_images[r,...])
        #    axs[i,1].imshow(_x[r,...])
        #    for j in range(2,NNEIGHBOURS+2):
        #        axs[i,j].imshow(neighbours[r,j-2,...])
        #        axs[i,j].set_title('N{} - {}'.format(j-1, round(neighbours_dist[r,j-2]),3),fontsize=5)
        #        axs[i,j].axis('off')
        #plt.savefig('/tmp/neighbours/n_{}_{}'.format(args.anomaly_class,epoch))
        #plt.close('all')
        ###################
        for image_batch in train_dataset:
            auto_loss  =  train_step(ae,image_batch)

        generate_and_save_images(ae,
                                 epoch + 1,
                                 image_batch[:25,...],
                                 'AE',
                                 args)
        save_checkpoint(ae,epoch, args,'AE','ae')

        ae_loss.append(auto_loss)

        print_epoch('AE',epoch,time.time()-start,{'AE Loss':auto_loss.numpy()},None)

    generate_and_save_training([ae_loss],
                                ['ae loss'],
                                'AE',args)
    generate_and_save_images(ae,epoch,image_batch[:25,...],'AE',args)

    return ae

def main(train_dataset,train_images,train_labels,test_images,test_labels, test_masks,args):
    ae = Autoencoder(args)
    ae = train(ae,train_dataset, train_images,test_images,test_labels,args)
    end_routine(train_images, test_images, test_labels, test_masks, [ae], 'AE', args)

    
if __name__  == '__main__':
    main()
