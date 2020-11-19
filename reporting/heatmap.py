import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')

def add_df_parameters(dataset):
    """
        Repeated from barplotter 
    """
    df = pd.read_csv('outputs/results_{}.csv'.format(dataset))
    index = 0
    data = []
    for i in range(df.shape[0]):
        d = df.iloc[i]
        scores = np.load('outputs/{}/{}/{}/latent_scores.pkl'.format(d.Model,
                                                                      d.Class,
                                                                      d.Name),
                                                                      allow_pickle=True)
        for s in scores:
            vals = scores[s]
            temp = [d.Model, d.Name, d.Latent_Dim,
                    d.Class, vals[0], vals[1],
                    d.AUC_Reconstruction_Error, vals[2],
                    d.F1_Reconstruction_Error, vals[3]]
            data.append(temp)

    df_new = pd.DataFrame(data,columns=df.columns)

    return df_new

def heatmap(dataset='MNIST',legend=True):
    """
        Creates heatmaps for K and r as shown in paper

        dataset (str): Name of the dataset 
        legend (bool): show the legend?
    """
    print('Dataset = {}\n'.format(dataset))
    df = add_df_parameters(dataset)

    df_agg = df.groupby(['Model',
                        'Class',
                        'Latent_Dim',
                        'Radius',
                        'Neighbour']).agg({'AUC_Reconstruction_Error': 'mean',
                                           'AUC_Latent_Error': 'mean'}).reset_index()
    r_heat_mx = np.zeros([5,10])#np.zeros([10,5])
    K_heat_mx = np.zeros([5,10])#np.zeros([10,5])
    vals = np.zeros([5,10])#np.zeros([10,5])


    fig,ax = plt.subplots(1,2,figsize=(6,1.5),sharex=True,sharey=True)

    for i,model in enumerate(pd.unique(df.Model)):
        for cl in np.sort(pd.unique(df.Class)):

            idx_latent = df_agg[(df_agg.Model == model) & 
                          (df_agg.Class == cl)].AUC_Latent_Error.idxmax()
            ld = df_agg.iloc[idx_latent]['Latent_Dim']
            rad  = df_agg.iloc[idx_latent]['Radius']
            neigh  = df_agg.iloc[idx_latent]['Neighbour']
            r_heat_mx[i,cl] = rad
            K_heat_mx[i,cl] = neigh
            vals[i,cl] = df_agg.iloc[idx_latent]['AUC_Latent_Error']
            print('Model {} for Digit {}  with Neighbours {}'.format(model,cl,neigh))


    ax[0].pcolor(r_heat_mx,vmin=1,vmax=30,edgecolor='w',linewidths=0.1)
    im = ax[1].pcolor(K_heat_mx,vmin=1,vmax=30,edgecolor='w',linewidths=0.2)

    #for axis in ax:
    #    for cl in range(10):
    #        for mod in range(5):
    #            text = axis.text(cl+0.5, mod+0.5, round(vals[mod,cl],2),
    #                           ha="center", va="center", color="w",fontsize=5,weight='bold')


    # Add xticks on the middle of the group bars
    mnist_ticks =['0','1', '2', '3', '4', '5','6','7','8','9']
    fmnist_ticks =['top','pants', 'jersey', 'dress', 'coat', 'sandal','shirt','sneaker','bag','boot']
    cifar_ticks =['plane','car', 'bird', 'cat', 'deer', 'dog','frog','horse','ship','truck']

    if dataset == 'MNIST' :
        ticks = mnist_ticks
        xlabel = 'MNIST Classes'
    elif dataset == 'FASHION_MNIST':
        ticks = fmnist_ticks
        xlabel = 'FMNIST Classes'
    elif dataset == 'CIFAR10':
        ticks = cifar_ticks
        xlabel = 'CIFAR-10 Classes'

    ax[0].set_title('$r$')
    ax[0].set_ylabel('Model')
    plt.sca(ax[0])
    plt.xticks(np.arange(0,10)+0.5, range(10))

    ax[1].set_title('$\mathit{K}$')
    plt.sca(ax[1])
    plt.xticks(np.arange(0,10)+0.5, range(10))
    plt.yticks(np.arange(0,5)+0.5, ['AE-con','AE-res','AE','VAE','AAE'])


    # Create legend & Show graphic
    cbar = fig.colorbar(im, ax=ax.ravel().tolist())#, shrink=0.95)
    cbar.set_ticks([1,15,30])
    cbar.set_ticklabels(['1', '50', '100'])

    plt.text(-1.5, -2.5,'MNIST Digit', weight='normal',ha='center')#, va='center')

    plt.savefig('outputs/{}_heatmaps.png'.format(dataset),bbox_inches='tight',dpi=300)
    plt.show()


