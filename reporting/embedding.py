import numpy as np 
import pandas as pd
import tensorflow as tf
from sklearn import metrics,neighbors
from random import sample,randint

import matplotlib
import matplotlib.image as mpimg
from matplotlib import image, pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data
import matplotlib.cm as cm

import sys

def plot_overlay(autoencoder, data):
    """
        @TODO update this doc string, make it work for VAE
        This function plots a 2d scatter plot with the data superimposed over each point
        autoencoder (keras.model) the autoencoder based model 
        data (np.array) the preprocessed training data that is in a list format. 
    """
    plt.rcParams['image.cmap'] = 'viridis'
    #embeddings  = autoencoder.reparameterize(z_mean, z_log_var)
    #z_mean, z_log_var = autoencoder.encoder(data,vae=True)
    #embeddings  = autoencoder.reparameterize(z_mean, z_log_var)
    embeddings =  autoencoder.encoder(data)
    x_hat = autoencoder(data).numpy()

    _data = data

    _x_hat = x_hat

    fig,ax = plt.subplots(1,2,figsize=(15,10));

    for x, y, image_path, output_path  in zip(embeddings[:,0], 
                                              embeddings[:,1], 
                                              _data[...,0],
                                              _x_hat[...,0]):

        imscatter(x, y, image_path, zoom=0.7, ax=ax[0]) 
        ax[0].title.set_text('Embedding overlaid with input');
        imscatter(x, y, output_path, zoom=0.7, ax=ax[1]) 
        ax[1].title.set_text('Embedding overlaid with output');

    plt.tight_layout()
    ax[0].grid();
    ax[1].grid();
    plt.suptitle('Scatter Plot of Embedding with Images Overlayed');
    plt.savefig('/tmp/em_temp.png',dpi=600)

def plot_knn_neighbours(model,data,labels,nnbour=5):
    #TODO expand this to other models
    fig,ax = plt.subplots(10,6,figsize=(15,7))
    z = model.encoder(data).numpy()

    nbrs = neighbors.NearestNeighbors(n_neighbors=nnbour, 
                                      algorithm='ball_tree',
                                      n_jobs=-1).fit(z) # using KNN
    knn_neighbours = nbrs.kneighbors(z,return_distance=False)#KNN

    for j,label in enumerate(pd.unique(labels)):
        indx = (np.where(labels== label)[0][0] )
    #    print('Error for feature {} is {}'.format(label,error[ns[indx]].shape))
        ax[j,0].imshow(data[indx,...,0]); ax[j,0].title.set_text('{}'.format(label))
        for i,n in enumerate(knn_neighbours[indx]):
            if i>4: continue
            image = model.decoder(np.expand_dims(z[n],axis=0))
            ax[j,i+1].imshow(image[0,...,0]); ax[j,i+1].title.set_text('Neighbour {}'.format(n))

    fig.tight_layout()
    plt.suptitle('KNN Neighbours of Each Unique label for NN = {}'.format(nnbour));
    plt.savefig('/tmp/knn_neighbours.png',dpi=600)

def plot_radius_neighbours(model,data,labels,rad=1.0):
    # expand this to other models
    fig,ax = plt.subplots(10,6,figsize=(15,7))
    z = model.encoder(data).numpy()

    nbrs = neighbors.NearestNeighbors(radius=rad,
                                      algorithm='ball_tree',
                                      n_jobs=-1).fit(z) 
    rad_neighbours = nbrs.radius_neighbors(z,return_distance=False)


    for j,label in enumerate(pd.unique(labels)):
        indx = (np.where(labels== label)[0][0] )
    #    print('Error for feature {} is {}'.format(label,error[ns[indx]].shape))
        ax[j,0].imshow(data[indx,...,0]); ax[j,0].title.set_text('{}'.format(label))
        for i,n in enumerate(rad_neighbours[indx]):
            if i>4: continue
            image = model.decoder(np.expand_dims(z[n],axis=0))
            ax[j,i+1].imshow(image[0,...,0]); ax[j,i+1].title.set_text('Neighbour {}'.format(n))

    fig.tight_layout()
    plt.suptitle('Rad Neighbours of Each Unique label for rad = {}'.format(rad));
    plt.savefig('/tmp/rad_neighbours.png',dpi=600)

def imscatter(x, y, image, ax=None, zoom=1):
    """
        code adapted from: https://gist.github.com/feeblefruits/20e7f98a4c6a47075c8bfce7c06749c2
    """
    if ax is None:
         ax = plt.gca()
    im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists
