import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors
from collections import OrderedDict
from glob import glob
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

def sensitivity_heatmap(dataset='MNIST',legend=True, multiple=False):
    """
        Creates heatmaps for K and r as shown in paper

        dataset (str): Name of the dataset 
        legend (bool): show the legend?
    """
    print('Dataset = {}\n'.format(dataset))

    df = None
    if multiple:
        for f in glob('outputs/results_{}_*'.format(dataset)):
            if ('updated' in f): continue

            print('Loading data from {}'.format(f[16:-4]))
            if df is None:
                df =  add_df_parameters(f[16:-4])
            else: df = df.append(add_df_parameters(f[16:-4]))

    mdl,ld = get_max_parameters(df) 
    print('Maximum NLN AUROC for mdl={}, ld={}'.format(mdl,ld))
    df = df[(df.Model == mdl)&(df.Latent_Dim ==ld)]

    df_agg = df.groupby(['Radius',
                        'Neighbour']).agg({'AUC_Latent_Error': 'mean'}).reset_index()

    rs = np.sort(pd.unique(df.Radius))
    Ks = np.sort(pd.unique(df.Neighbour))
    mx = np.zeros([len(Ks), 
                   len(rs)])

    fig,ax = plt.subplots(1,1,figsize=(3,1.5))
    for i,r in enumerate(rs):
        for j,K in enumerate(Ks):
            mx[i,j] = df_agg[(df_agg.Radius == r) & (df_agg.Neighbour == K)].AUC_Latent_Error

    
    im = ax.pcolor(mx,vmin=np.min(mx),vmax=np.max(mx),edgecolor='w',linewidths=0.1)

    ax.set_title('{}: Model = {} and L= {}'.format(dataset,mdl,ld),fontsize=8)
    ax.set_ylabel('Radius ($r$)')
    ax.set_xlabel('Neighbours ($K$)')

    # Major ticks
    ax.set_xticks(np.arange(0.5, len(Ks), 1))
    ax.set_yticks(np.arange(0.5, len(rs), 1))


    # Labels for major ticks
    ax.set_xticklabels(Ks)
    ax.set_yticklabels(rs)

    # Create legend & Show graphic
    cbar = fig.colorbar(im, ax=ax)#, shrink=0.95)
    cbar.ax.set_title('AUROC',fontsize=3)

    #cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    #cbar.set_ticklabels(['0', '0.25', '0.5', '0.75', '1.0'])

    plt.savefig('outputs/{}_heatmaps.png'.format(dataset),bbox_inches='tight',dpi=300)
    plt.show()

def sensitivity_heatmap_shared(legend=True, multiple=True):
    """
        Creates heatmaps for K and r as shown in paper

        dataset (str): Name of the dataset 
        legend (bool): show the legend?
    """
    datasets = ['MNIST', 'FASHION_MNIST','CIFAR10','MVTEC']
    fig,axs = plt.subplots(2,2,figsize=(6,3),sharex=True,sharey=True)
    x_inds, y_inds  = [0,0,1,1], [0,1,0,1]
    for dataset,x,y in zip(datasets,x_inds,y_inds):
        df = None
        for f in glob('outputs/results_{}_*'.format(dataset)):
            if ('updated' in f): continue

            print('Loading data from {}'.format(f[16:-4]))
            if df is None:
                df =  add_df_parameters(f[16:-4])
            else: df = df.append(add_df_parameters(f[16:-4]))

        mdl,ld = get_max_parameters(df) 
        print('Maximum NLN AUROC for mdl={}, ld={}'.format(mdl,ld))
        df = df[(df.Model == mdl)&(df.Latent_Dim ==ld)]

        df_agg = df.groupby(['Radius',
                            'Neighbour']).agg({'AUC_Latent_Error': 'mean'}).reset_index()

        rs = np.sort(pd.unique(df.Radius))
        Ks = np.sort(pd.unique(df.Neighbour))
        mx = np.zeros([len(Ks), 
                       len(rs)])

        for i,r in enumerate(rs):
            for j,K in enumerate(Ks):
                mx[i,j] = df_agg[(df_agg.Radius == r) & (df_agg.Neighbour == K)].AUC_Latent_Error

        
        im = axs[x,y].pcolor(mx,vmin=np.min(mx),vmax=np.max(mx),edgecolor='w',linewidths=0.1)

        axs[x,y].set_title('{}'.format(dataset),fontsize=8)

        # Major ticks
        axs[x,y].set_xticks(np.arange(0.5, len(Ks), 1))
        axs[x,y].set_yticks(np.arange(0.5, len(rs), 1))


        # Labels for major ticks
        axs[x,y].set_xticklabels(Ks)
        axs[x,y].set_yticklabels(rs)

        # Create legend & Show graphic
        cbar = fig.colorbar(im, ax=axs[x,y])#, shrink=0.95)


    fig.text(0.5, 0.00, 'Radius ($r$)', ha='center')
    fig.text(0.00, 0.5, 'Neighbours ($K$)', va='center', rotation='vertical')
    plt.tight_layout()

    plt.savefig('outputs/shared_sensitivity_heatmaps.png',bbox_inches='tight',dpi=300)
    plt.show()

def get_max_parameters(df):
    """
        Function that receives df and returns the average maximum model and latent_dim 

        df (pd.DataFrame) dataframe of joined results
    """

    df_agg = df.groupby(['Model',
                        'Latent_Dim',
                        'Radius',
                        'Neighbour']).agg({'AUC_Latent_Error': 'mean'}).reset_index()

    idx= df_agg.AUC_Latent_Error.idxmax()
    model = df_agg.iloc[idx].Model
    ld = df_agg.iloc[idx].Latent_Dim

    return  model,ld

