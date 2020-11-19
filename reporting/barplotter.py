import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')

def add_df_parameters(dataset):
    """
        Adds the missing parameters to the results.csv file 

        dataset (str)  name of the preexisting dataset
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

def barplot(dataset,legend=True):
    """
        Creates a single barplot for a given dataset 

        dataset (str) name of dataset  
    """
    print('Dataset = {}\n'.format(dataset))
    df = add_df_parameters(dataset)

    df_agg = df.groupby(['Model',
                        'Class',
                        'Latent_Dim',
                        'Radius',
                        'Neighbour']).agg({'AUC_Reconstruction_Error': 'mean',
                                           'AUC_Latent_Error': 'mean'}).reset_index()
    _max = len(pd.unique(df.Class))
    recon_width = 0.35
    latent_width = 0.10

    if legend: x = 3
    else: x = 0
    fig,ax = plt.subplots(1,1,figsize=(10+x,8+x/3))

    colors = cm.tab20b(np.linspace(0, 1, 10))

    colours_recon = []#'#750256','#6B1C8C','#21908D','#5AC865','#FFFF21']
    colours_latent =[]#'#450256','#3B1C8C','#51908D','#8AC865','#F9E721']

    for c in range(0,10,2):
        colours_recon.append(colors[c+1])
        colours_latent.append(colors[c])
    plt.rcParams['hatch.color'] = 'red'

    r = np.arange(0,2*_max,2)
    for model,color_recon,color_latent in zip(pd.unique(df.Model),
                                              colours_recon,
                                              colours_latent):
        recon,latent = [],[]
        for cl in np.sort(pd.unique(df.Class)):
            idx_latent = (df_agg[df_agg.Model == model].AUC_Latent_Error.idxmax())
            ld = df_agg.iloc[idx_latent]['Latent_Dim']
            rad  = df_agg.iloc[idx_latent]['Radius']
            neigh  = df_agg.iloc[idx_latent]['Neighbour']

            df_max = df[(df.Model == model) &
                        (df.Latent_Dim == ld) &
                        (df.Neighbour == neigh) &
                        (df.Radius == rad) &
                        (df.Class == cl)]

            latent.append(df_max['AUC_Latent_Error'].values[0])

            recon.append(df_max['AUC_Reconstruction_Error'].values[0])


        print('Model {} \nMean Reconstruction error {} \nMean Latent Error {}\nPercetange Increase {}\n'.format(model,np.mean(recon), np.mean(latent), round(1-np.mean(recon)/np.mean(latent),4)))
        print('Latent Dim: {}'.format(ld))

        r = [x + recon_width for x in r]

        if model == 'GANomaly': 
            name = 'AE-con'
        elif ((model == 'DAE') or (model == 'DAE_disc')): 
            name = 'AE-res'
        else: 
            name = model

        ax.bar(r,
               recon ,
               width=recon_width,
               color=color_recon,
               edgecolor='white',
               label='{}'.format(name))

        ax.bar(r,
               latent,
               width=latent_width,
               color=color_latent,
               hatch='x',
               #edgecolor='black',
               label='{}: NLN '.format(name))


    # Add xticks on the middle of the group bars
    r = [x + recon_width for x in r]
    mnist_ticks =['0','1', '2', '3', '4', '5','6','7','8','9']
    fmnist_ticks =['top','pants', 'jersey', 'dress', 'coat', 'sandal','shirt','sneaker','bag','boot']
    cifar_ticks =['plane','car', 'bird', 'cat', 'deer', 'dog','frog','horse','ship','truck']

    if dataset == 'MNIST' :
        ticks = mnist_ticks
        xlabel = 'MNIST Classes'
    elif dataset == 'FMNIST':
        ticks = fmnist_ticks
        xlabel = 'FMNIST Classes'
    elif dataset == 'CIFAR10':
        ticks = cifar_ticks
        xlabel = 'CIFAR-10 Classes'

    plt.xticks([2*r + 3*recon_width for r in range(_max)], ticks)
    ax.set_ylim([0,1])
    plt.xlabel(xlabel)
    plt.ylabel('AUROC Score')

    # Create legend & Show graphic
    if legend:
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),prop={'size': 17}, ncol=5)
#    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),ncol=2,fancybox=True,shadow=True)

    plt.tight_layout()
    plt.savefig('outputs/{}_barplot.png'.format(dataset),dpi=300)
    plt.show()

def barplot_shared():
    """
        Creates shared barplot for MNIST and CIFAR 
    """
    fig,axs = plt.subplots(1,2,figsize=(14,4))

 
    colors = cm.tab20b(np.linspace(0, 1, 10))

    colours_recon = []#'#750256','#6B1C8C','#21908D','#5AC865','#FFFF21']
    colours_latent =[]#'#450256','#3B1C8C','#51908D','#8AC865','#F9E721']
    for c in range(0,10,2):
        colours_recon.append(colors[c+1])
        colours_latent.append(colors[c])

    recon_width = 0.35
    latent_width = 0.10

    #plt.rcParams['hatch.linewidth'] = 3
    plt.rcParams['hatch.color'] = 'red'

    for dataset,ax in zip(['MNIST','CIFAR10'],axs):
        df = add_df_parameters(dataset)
        _max = len(pd.unique(df.Class))
        r = np.arange(0,2*_max,2)

        df_agg = df.groupby(['Model',
                            'Class',
                            'Latent_Dim',
                            'Radius',
                            'Neighbour']).agg({'AUC_Reconstruction_Error': 'mean', 
                                                'AUC_Latent_Error': 'mean'}).reset_index()

        for model,color_recon,color_latent in zip(pd.unique(df.Model),
                                                  colours_recon,
                                                  colours_latent):

            idx_latent = (df_agg[df_agg.Model == model].AUC_Latent_Error.idxmax())
            ld = df_agg.iloc[idx_latent]['Latent_Dim']
            rad = df_agg.iloc[idx_latent]['Radius']
            neigh = df_agg.iloc[idx_latent]['Neighbour']
            latent = list(df[(df.Model == model)  & 
                             (df.Latent_Dim == ld) &
                             (df.Radius== rad) &
                             (df.Neighbour== neigh)]['AUC_Latent_Error'])

            recon = list(df[(df.Model == model)  & 
                            (df.Latent_Dim == ld) &
                            (df.Radius== rad) &
                            (df.Neighbour== neigh)]['AUC_Reconstruction_Error'])
    #        print('Mean Reconstruction error {} \nMean Latent Error {}\n Percetange Increase {}\n'.format(np.mean(recon), np.mean(latent), round(1-np.mean(recon)/np.mean(latent),4)))

            r = [x + recon_width for x in r]

            if model == 'GANomaly': name = 'AE-con'

            elif (model == 'DAE') or (model == 'DAE_disc'): 
                name = 'AE-res'

            else: name = model
            
            ax.bar(r, 
                   recon , 
                   width = recon_width, 
                   color=color_recon, 
                   edgecolor='white',
                   label='{}'.format(name))

            ax.bar(r, 
                   latent, 
                   width = latent_width, 
                   color=color_latent, 
                   hatch='x',
                   label='{}: NLN '.format(name))
        
         
        # Add xticks on the middle of the group bars
        r = [x + recon_width for x in r]
        mnist_ticks =['0','1', '2', '3', '4', '5','6','7','8','9']
        fmnist_ticks =['top','pants', 'jersey', 'dress', 'coat', 'sandal','shirt','sneaker','bag','boot']
        cifar_ticks =['plane','car', 'bird', 'cat', 'deer', 'dog','frog','horse','ship','truck'] 

        if dataset == 'MNIST' :
            ticks = mnist_ticks
            xlabel = 'MNIST Classes'
            rot=30
        elif dataset == 'FASHION_MNIST':
            ticks = fmnist_ticks
            xlabel = 'FMNIST Classes'
            rot = 30
        elif dataset == 'CIFAR10': 
            ticks = cifar_ticks
            xlabel = 'CIFAR-10 Classes'
            rot = 30

        ax.set_xticks([2*r + 3*recon_width for r in range(10)])
        ax.set_xticklabels(ticks,rotation=rot)
        ax.set_ylim([0,1])
        ax.set_xlabel(xlabel) 
        ax.set_ylabel('AUROC Score') 
     
    # Create legend & Show graphic
    plt.legend(loc='upper center', bbox_to_anchor=(-0.1, -0.25),prop={'size': 15}, ncol=5)

    #plt.tight_layout()
    plt.savefig('outputs/joined_result.png',bbox_inches='tight',dpi=300)
    plt.show()

