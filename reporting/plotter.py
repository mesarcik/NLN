import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm

def plot_results():
    df = pd.read_csv('outputs/results.csv')
    x = list(pd.unique(df.Class))
    model_types  = list(pd.unique(df.Model))

    colours=cm.rainbow(np.linspace(0,1,len(model_types)))

    fig, ax= plt.subplots(1,1,figsize=(13,7))
    for model_type,colour in zip(model_types,colours):
        plt.plot(x,
                 list(df.AUC_Reconstruction_Error[df.Model==model_type]),
                 label='{} Original'.format(model_type),
                 linewidth = 2,
                 color=colour)

        plt.plot(x,
                list(df.AUC_Latent_Error[df.Model==model_type]),
                label='{} Latent'.format(model_type), 
                linewidth=6,
                color=colour)

    plt.legend()
    plt.grid()
    plt.title('AUC Scores for Latent space of 20 for MNIST')
    plt.xlabel('MNIST Digit')
    plt.ylabel('AUC-score')
    plt.tight_layout()
    plt.savefig('outputs/f1_scores.png')
    plt.close('all')

    fig, ax= plt.subplots(1,1,figsize=(13,7))
    for model_type,colour in zip(model_types,colours):
        plt.plot(x,
                 df.AUC_Reconstruction_Error[df.Model==model_type],
                 label='{} Original'.format(model_type),
                 linewidth = 2,
                 color=colour)
        plt.plot(x,
                 df.AUC_Latent_Error[df.Model==model_type],
                 label='{} Latent'.format(model_type), 
                 linewidth=6,
                 color=colour)
    plt.legend()
    plt.grid()
    plt.title('ROC_AUC Scores for Latent space of 20 for MNIST')
    plt.xlabel('MNIST Digit')
    plt.ylabel('AUC-score')
    plt.tight_layout()
    plt.savefig('outputs/AUC_scores.png')

def plot_per_model():
    df = pd.read_csv('outputs/results.csv')
    x = list(pd.unique(df.Class))
    model_types  = list(pd.unique(df.Model))
    latent_dims = list(pd.unique(df.Latent_Dim))


    colours=cm.rainbow(np.linspace(0,1,len(latent_dims)))

    for model_type in model_types:
        fig, ax= plt.subplots(1,1,figsize=(13,7))
        for latent_dim,colour  in zip(latent_dims,colours):
            if len(x) !=  len(list(df.AUC_Reconstruction_Error[(df.Model==model_type) & (df.Latent_Dim==latent_dim)])):
                break
            plt.plot(x,
                     list(df.AUC_Reconstruction_Error[(df.Model==model_type) & (df.Latent_Dim==latent_dim)]),
                     label='{} Original'.format(latent_dim),
                     linewidth = 2,
                     color=colour)

            plt.plot(x,
                    list(df.AUC_Latent_Error[(df.Model==model_type) & (df.Latent_Dim==latent_dim)]),
                    '--',
                    label='{} Latent'.format(latent_dim), 
                    linewidth=2,
                    color=colour)

        plt.legend()
        plt.grid()
        plt.title('AUC Scores for {} for MNIST'.format(model_type))
        plt.xlabel('MNIST Digit')
        plt.ylabel('AUC-score')
        plt.tight_layout()
        plt.savefig('outputs/f1_scores_{}.png'.format(model_type))
        plt.close('all')

    for model_type in model_types:
        fig, ax= plt.subplots(1,1,figsize=(13,7))
        for latent_dim,colour  in zip(latent_dims,colours):
            plt.plot(x,
                     df.AUC_Reconstruction_Error[(df.Model==model_type) & (df.Latent_Dim==latent_dim)],
                     label='{} Original'.format(latent_dim),
                     linewidth = 2,
                     color=colour)
            plt.plot(x,
                     df.AUC_Latent_Error[(df.Model==model_type) & (df.Latent_Dim==latent_dim)],
                     '--',
                     label='{} Latent'.format(latent_dim), 
                     linewidth=2,
                     color=colour)
        plt.legend()
        plt.grid()
        plt.title('ROC_AUC Scores for {} for MNIST'.format(model_type))
        plt.xlabel('MNIST Digit')
        plt.ylabel('ROC_AUC-score')
        plt.tight_layout()
        plt.savefig('outputs/AUC_scores_{}.png'.format(model_type))
        plt.close('all')

def plot_max():
    df = pd.read_csv('outputs/results.csv')
    model_types  = list(pd.unique(df.Model))
    digits = list(pd.unique(df.Class))

    colours=cm.rainbow(np.linspace(0,1,len(model_types)))
    fig, ax= plt.subplots(1,1,figsize=(13,7))
    for model_type,colour in zip(model_types,colours):
        fig_n, ax_n= plt.subplots(1,1,figsize=(13,7))

        max_latent,max_orig  = [],[]
        max_neighbour_latent,max_dim_latent = [],[]
        max_neighbour_recon, max_dim_recon = [],[]

        for digit in digits:
            max_orig.append(max(df.AUC_Reconstruction_Error[(df.Model==model_type) &
                                (df.Class == digit)]))

            max_latent.append(max(df.AUC_Latent_Error[(df.Model==model_type) &
                                (df.Class == digit)]))
        ax.plot(digits,
                 max_orig,
                 label='Max Reconstruction Error {}'.format(model_type),
                 #linewidth=3,
                 color=colour)

        ax.plot(digits,
                 max_latent,
                 '--',
                 label='Max Latent Error {}'.format(model_type), 
                 #linewidth=3,
                 color=colour)

        print('For model {}\nmax latent = {},\nmax Reconstruction = {}\n'.format(model_type,max_latent,max_orig))
    ax.legend()
    ax.grid()
    ax.title.set_text('MAX AUC Scores for MNIST')
    ax.set_xlabel('MNIST Digit')
    ax.set_ylabel('ROC_AUC-score')
    ax.set_ylim([0,1])
    plt.tight_layout()
    fig.savefig('outputs/MAX_AUC_scores.png')
    plt.close(fig)

def plot_max_avg():
    df = pd.read_csv('outputs/results.csv')
    #df_agg = df.groupby(['Model','Latent_Dim','Neighbour']).agg({'AUC_Reconstruction_Error': 'mean', 'AUC_Latent_Error':'mean'}).reset_index()
    df_agg = df.groupby(['Model','Radius','Latent_Dim','Neighbour']).agg({'AUC_Reconstruction_Error': 'mean', 'AUC_Latent_Error':'mean'}).reset_index()
    fig0,ax = plt.subplots(1,1,figsize=(13,7))
    
    model_types  = list(pd.unique(df.Model))
    colours=cm.rainbow(np.linspace(0,1,len(model_types)))

    for model,colour in zip(pd.unique(df_agg.Model),colours):
        fig = plt.figure()
        idx_recon = (df_agg[df_agg.Model == model].AUC_Reconstruction_Error.idxmax())
        idx_latent = (df_agg[df_agg.Model == model].AUC_Latent_Error.idxmax())

        temp_recon = df_agg.iloc[idx_recon]
        temp_latent = df_agg.iloc[idx_latent]

        temp = df[(df.Model == temp_recon.Model) &
                   (df.Latent_Dim == temp_recon.Latent_Dim)]
        plt.plot(temp['Class'],
                 temp['AUC_Reconstruction_Error'],
                 label='Reconstruction Error for latent space of {}'.format(temp_recon.Latent_Dim))
        print('Best average reconstruction error for model {} is {}'.format(model,np.mean(temp['AUC_Reconstruction_Error'])))

        ax.plot(temp['Class'],
                 temp['AUC_Reconstruction_Error'],
                 color=colour,
                 label='{}: Standard'.format(model))

        temp = df[(df.Model == temp_latent.Model) &
                   (df.Latent_Dim == temp_latent.Latent_Dim)]
        plt.plot(temp['Class'],
                 temp['AUC_Latent_Error'],
                 '--',
                 #label='Latent Error for latent space of {}  with {} Neighbour'.format(temp_latent.Latent_Dim,temp_latent.Neighbour))
                 label='Latent Error for latent space of {} and Radius of {} with {} Neighbour'.format(temp_latent.Latent_Dim,temp_latent.Radius,temp_latent.Neighbour))


        print('Best average latent error for model {} is {}'.format(model,np.mean(temp['AUC_Latent_Error'])))

        print('{}: Percentage increase = {} \n'.format(model,1 - np.mean(temp['AUC_Latent_Error']) / np.mean(temp['AUC_Reconstruction_Error'])))

        ax.plot(temp['Class'],
                 temp['AUC_Latent_Error'],
                 '--',
                 color=colour,
                 label='{}: Latent Error'.format(model))

        plt.legend()
        plt.grid()
        plt.title('Max Reconstruction Error For Latent Dim for model {}'.format(model))
        plt.xlabel('MNIST Digit')
        plt.ylabel('AUC-score')
        ax1 = plt.gca()
        ax1.set_ylim([0,1])
        plt.tight_layout()
        fig.savefig('outputs/MAX_AUC_Scores_{}.png'.format(model))
        plt.close(fig)

    ax.legend()
    ax.grid()
    ax.set_title('Max Reconstruction Error For Latent Dim')
    ax.set_xlabel('MNIST Digit')
    ax.set_ylabel('AUC-score')
    ax.set_ylim([0,1])
    fig0.tight_layout()
    fig0.savefig('outputs/MAX_AUC_Scores_overlaid.png')

