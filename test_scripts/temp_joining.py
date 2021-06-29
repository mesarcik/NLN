import tensorflow as tf
import numpy as np
from utils.data import *
from data import *
from models import Autoencoder
from utils.metrics.latent_reconstruction import * 
from matplotlib import pyplot as plt

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

args = Namespace(input_shape=(32,32,3),
                  rotate=False,
                  patches=True,
                  patch_x = 8,
                  patch_y=8,
                  latent_dim = 500,
                  patch_stride_x = 8,
                  patch_stride_y = 8,
                  # NLN PARAMS 
                  anomaly_class= 0, 
                  neighbours = [5],
                  radius = [10.0],
                  algorithm = 'knn')

def plot_sep(test_images,n_patches,x_out,masks_patches,x_hat):
    # plot 3 plots, input, patches, patches out 
    fig, axs = plt.subplots(1,1)
    axs.imshow(test_images[1,...])
    fig.suptitle('Input image')
    fig.savefig('/tmp/0_input')
    plt.close(fig)

    #input patches 
    fig1, axs1 = plt.subplots(n_patches,n_patches)
    fig2, axs2 = plt.subplots(n_patches,n_patches)
    fig3, axs3 = plt.subplots(n_patches,n_patches)
    fig4, axs4 = plt.subplots(n_patches,n_patches)
    fig5, axs5 = plt.subplots(n_patches,n_patches)

    counter = n_patches*n_patches
    for i in range(n_patches):
        for j in range(n_patches):
            axs1[i,j].imshow(x_out[counter,...],vmin=0, vmax=1)
            axs1[i,j].xaxis.set_visible(False)
            axs1[i,j].yaxis.set_visible(False)

            axs2[i,j].imshow(masks_patches[counter,...], vmin=0, vmax=1)
            axs2[i,j].xaxis.set_visible(False)
            axs2[i,j].yaxis.set_visible(False)

            axs3[i,j].imshow(x_hat[counter,...], vmin=0, vmax=1)
            axs3[i,j].xaxis.set_visible(False)
            axs3[i,j].yaxis.set_visible(False)

            er = np.abs(x_out[counter,...] - x_hat[counter,...])
            axs4[i,j].imshow(er ,vmin=0, vmax=1)
            axs4[i,j].xaxis.set_visible(False)
            axs4[i,j].yaxis.set_visible(False)

            er=np.mean(er,axis=-1)
            er = er>0.05
            axs5[i,j].imshow(er.astype('int8'), vmin =0, vmax=1)
            axs5[i,j].xaxis.set_visible(False)
            axs5[i,j].yaxis.set_visible(False)
            counter +=1

    fig1.suptitle('Input patches')
    fig1.savefig('/tmp/1_patches')
    plt.close(fig1)

    fig2.suptitle('Mask patches')
    fig2.savefig('/tmp/2_masks')
    plt.close(fig2)

    fig3.suptitle('Output  patches')
    fig3.savefig('/tmp/3_output')
    plt.close(fig3)

    fig4.suptitle('Error patches')
    fig4.savefig('/tmp/4_error')
    plt.close(fig4)

    fig5.suptitle('Error patches thresholded mean>0.25')
    fig5.savefig('/tmp/5_error_thr')
    plt.close(fig5)

def plot_recon(test_images, x_out, mean_neighbours,  masks_patches,x_hat,args,ind=0):
    # plot 3 plots, input, patches, patches out 
    FIG, AXS = plt.subplots(1,7,figsize=(10, 5))

    fig, axs = plt.subplots(1,1)
    axs.imshow(test_images[ind,...])

    fig.suptitle('Input image')
    fig.savefig('/tmp/0_input')
    plt.close(fig)
    AXS[0].imshow(test_images[ind,...])
    AXS[0].set_title('Input')


    fig, axs = plt.subplots(1,1)
    recon = reconstruct(masks_patches,test_images,args)[...,0]
    axs.imshow(recon[ind,...],vmin=0, vmax=1)
    fig.suptitle('Mask patches')
    fig.savefig('/tmp/1_masks')
    plt.close(fig)
    AXS[1].imshow(recon[ind,...], vmin=0, vmax= 1)
    AXS[1].set_title('Mask patches')

    fig, axs = plt.subplots(1,1)
    recon_in = reconstruct(x_out,test_images,args)
    axs.imshow(recon_in[ind,...],vmin=0, vmax=1)
    fig.suptitle('Joined paches image')
    fig.savefig('/tmp/2_patches')
    plt.close(fig)
    AXS[2].imshow(recon_in[ind,...], vmin=0, vmax= 1)
    AXS[2].set_title('Joined patches')

    fig, axs = plt.subplots(1,1)
    recon_out = reconstruct(x_hat,test_images,args)
    axs.imshow(recon_out[ind,...],vmin=0, vmax=1)
    fig.suptitle('Output patches')
    fig.savefig('/tmp/3_output')
    plt.close(fig)
    AXS[3].imshow(recon_out[ind,...], vmin=0, vmax= 1)
    AXS[3].set_title('Output patches')

    fig, axs = plt.subplots(1,1)
    recon_out = reconstruct(mean_neighbours,test_images,args)
    axs.imshow(recon_out[ind,...],vmin=0, vmax=1)
    fig.suptitle('NLN Images')
    fig.savefig('/tmp/4_NLN')
    plt.close(fig)
    AXS[4].imshow(recon_out[ind,...], vmin=0, vmax= 1)
    AXS[4].set_title('NLN Images')

    fig, axs = plt.subplots(1,1)
    er = recon_in[ind,...] - recon_out[ind,...] 
    axs.imshow(er,vmin=0, vmax=1)
    fig.suptitle('Error patches')
    fig.savefig('/tmp/5_error')
    plt.close(fig)
    AXS[5].imshow(er, vmin=0, vmax= 1)
    AXS[5].set_title('Error patches')

    fig, axs = plt.subplots(1,1)
    er=np.mean(er,axis=-1)
    er = np.logical_or(er>0.05, er< -0.05) 
    axs.imshow(er,vmin=0, vmax=1)
    fig.suptitle('Error patches thresholded mean>0.05')
    fig.savefig('/tmp/6_error_thr')
    plt.close(fig)
    AXS[6].imshow(er, vmin=0, vmax= 1)
    AXS[6].set_title('Error patches threshodled >0.05')
    
    FIG.tight_layout()
    FIG.savefig('/tmp/6_joined_plot')

def reconstruct(x_out,test_images,args):
    t = x_out.transpose(0,2,1,3)
    n_patches = test_images.shape[1]//args.patch_x
    recon = np.empty([x_out.shape[0]//n_patches**2, args.patch_x*n_patches,args.patch_y*n_patches,x_out.shape[-1]])

    start, counter, indx, b  = 0, 0, 0, []

    for i in range(n_patches, x_out.shape[0], n_patches):
        b.append(np.reshape(np.stack(t[start:i,...],axis=0),(n_patches*args.patch_x,args.patch_x,x_out.shape[-1])))
        start = i
        counter +=1
        if counter == n_patches:
            recon[indx,...] = np.hstack(b)
            indx+=1
            counter, b = 0, []

    return recon

def main():
    # Load data 
    (train_images, train_labels), (test_images, test_labels, test_masks) = get_mvtec_images(args.anomaly_class,directory='/tmp/')

    test_images = test_images[test_labels == 'hazelnut']
    test_masks = test_masks[test_labels == 'hazelnut']
    test_labels = test_labels[test_labels == 'hazelnut']

    # Load model 
    ae = Autoencoder(args)
    ae.load_weights('/tmp/AE/training_checkpoints/checkpoint_full_model_ae')

    # For each image get model output its patches
    # probably need to figure out a way to automatically determine the patches.

    #test_images = process(test_images)
    test_masks = np.expand_dims(test_masks,axis=-1)
    train_images = process(train_images)
    test_images = process(test_images)

    x_out, y_out = get_patches(x = test_images, 
                               y= test_labels, 
                               p_size = (1, args.patch_x, args.patch_y, 1),
                               s_size = (1, args.patch_x, args.patch_y, 1),
                               rate = (1, 1, 1, 1),
                               padding = 'VALID')
#    x_out = process(x_out)

    train_x_out, train_y_out = get_patches(x = train_images, 
                                           y= train_labels, 
                                           p_size = (1, args.patch_x, args.patch_y, 1),
                                           s_size = (1, args.patch_x, args.patch_y, 1),
                                           rate = (1, 1, 1, 1),
                                           padding = 'VALID')
#    train_x_out = process(train_x_out)

    masks_patches, _ = get_patches(test_masks,
                                   test_labels,
                                   p_size = (1, args.patch_x, args.patch_y, 1),
                                   s_size = (1, args.patch_stride_x, args.patch_stride_y, 1),
                                   rate = (1, 1, 1, 1),
                                   padding = 'VALID')

    n_patches = test_images.shape[1]//args.input_shape[0]

    x_hat = ae(x_out[0:5000,...].astype('float32')).numpy()

    x_hat_train = ae(train_x_out[0:5000,...].astype('float32')).numpy()

    z_query,error_query = get_error('AE', ae, x_out[0:5000], return_z = True)
    z,_ = get_error('AE', ae, train_x_out[0:5000,...], return_z = True)

    nbrs = neighbors.NearestNeighbors(radius=args.radius[0], 
                                      algorithm='ball_tree',
                                      n_jobs=-1).fit(z) # using radius

    _,neighbours_idx =  nbrs.radius_neighbors(z_query,
                                              return_distance=True,
                                              sort_results=True)#radius

    error = []
    neighbours = [] 
#    neigh_img 
    z = np.zeros(x_hat_train.shape)
    for i,n in enumerate(neighbours_idx):
        if len(n) ==0: 
            temp = np.array([255])
            d = 255*np.ones(x_hat_train[0:1,...].shape)

        elif len(n) > args.neighbours[0]: 
            temp  = n[:args.neighbours[0]] 
            d = x_hat_train[temp.astype(int)]
        else: 
            temp  = n
            d = x_hat_train[temp.astype(int)]

        neighbours.append(np.mean(d,axis=0))
        im = np.stack([x_out[i]]*temp.shape[-1],axis=0)

        error.append(np.mean(np.abs(d - im)**2))

    mean_neighbours = np.array(neighbours)
    #plot(test_images,n_patches,x_out,masks_patches,x_hat)
    
    plot_recon(test_images, x_out, mean_neighbours, masks_patches,x_hat,args,ind=0)

if __name__ == '__main__':
    main()
