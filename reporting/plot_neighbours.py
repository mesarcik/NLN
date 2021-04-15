import tensorflow as tf
import numpy as np
from sklearn import metrics 
from matplotlib import pyplot as plt

N_PATCHES = 10

def plot_neighs(test_images, test_labels, test_masks, x_hat, neighbours, neighbours_dist, args):
    rs = np.random.randint(0,len(test_labels),N_PATCHES) 
    fig, ax = plt.subplots(N_PATCHES, neighbours.shape[1]+3, figsize=(10,10))

    if (('grid' in args.anomaly_class) or ('screw' in args.anomaly_class) or ('zipper' in args.anomaly_class)): 
        test_images = test_images[,...0] 
        x_hat = x_hat[,...0] 
        neighbours= neighbours[,...0] 

    for i,r in enumerate(rs):
        col = 0

        ax[i,col].imshow(test_images[r,...]); 
        ax[i,col].set_title('Input {} idx {}'.format(test_labels[r], r), fontsize=6)
        col+=1

        ax[i,col].imshow(test_masks[r,...]); 
        ax[i,col].set_title('Masks', fontsize=6)
        col+=1

        ax[i,col].imshow(x_hat[r,...]); 
        ax[i,col].set_title('Reconstruction', fontsize=6)
        col+=1

        for n, dist in zip(neighbours[r], neighbours_dist[r]): 
            ax[i,col].imshow(n)
            ax[i,col].set_title('neigh {}, dist {}'.format(col, round(dist,2)), fontsize=5)
            col+=1


    plt.tight_layout()
    plt.savefig('/tmp/neighbours/{}_neighbours'.format(args.anomaly_class))
    plt.close('all')



