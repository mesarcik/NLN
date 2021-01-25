# Script to read in all the results from the 3 different runs and report variance etc 
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


from utils.metrics import nearest_error
#from model_loader import load_model
from data import *

params = {'MNIST':['GANomaly', 50, 20, 5], #dataset, ['model', L, r, K]
          'CIFAR10':['AE',100,10,2],
          'FASHION_MNIST':['DAE_disc',100,20,5]}

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


class Namespace:
    """ 
        A hack to emulate input arguments
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def describe(dataset):
    """
        Calculates min, max, mean and var of a given set of results

    """
    df_output = pd.DataFrame(columns = ['Model', 
                                        'Name', 
                                        'Latent_Dim',
                                        'Class', 
                                        'Neighbour',
                                        'Radius',
                                        'AUC_Reconstruction_Error',
                                        'AUC_Latent_Error'])

    train_images, train_labels, test_images, test_labels = load_data(dataset)
    # For each dataset 
    for f in glob('outputs/results*'):
        if (dataset.split('_')[0] != f.split('_')[1]) or ('updated' in f): continue  # a hack for FASHION_MNIST
        print(f)
        df = pd.read_csv(f) 
        
        df = df[(df.Model == params[dataset][0]) &
                (df.Latent_Dim == params[dataset][1])].reset_index() #

        df.drop(['index','F1_Reconstruction_Error','F1_Latent_Error'],axis=1, inplace=True)

        for i in tqdm(range(df.shape[0])):
            entry = df.iloc[i]
            model_path = 'outputs/{}/{}/{}/training_checkpoints'.format(entry.Model,
                                                                              entry.Class,
                                                                              entry.Name)

            model = load_model(model_path, Namespace(latent_dim = entry.Latent_Dim, 
                                                    input_shape = train_images[0,:].shape))
            args = Namespace(neighbors = [params[dataset][3]], 
                             algorithm = 'radius',
                             anomaly_class = entry.Class,
                             radius = [params[dataset][2]],
                             model_name = entry.Name)

            max_auc,max_f1,max_neighbours,max_radius = nearest_error(model, 
                                                                     train_images[entry.Class != train_labels],
                                                                     test_images,
                                                                     test_labels, 
                                                                     entry.Model, 
                                                                     args, 
                                                                     False)

            df['AUC_Latent_Error'].iloc[i] =  max_auc
            df['Radius'].iloc[i] = params[dataset][2] 
            df['Neighbour'].iloc[i] = params[dataset][3] 
            df_output = df_output.append(df.iloc[i],ignore_index=True)


    df_output.to_csv('outputs/results_{}_updated_{}_{}_{}.csv'.format(dataset,
                                                                      params[dataset][0],
                                                                      params[dataset][1],
                                                                      params[dataset][2]),index=False)

    return df_output

def get_statistics(dataset = 'MNIST'):
    """
        Gives the mean and variance for a given dataset for multiple experiments
    """


    df_output = pd.DataFrame(columns= ['Model',
                                      'Class',
                                      'NLN_Latent_Dim',
                                      'Reconstruction_Latent_Dim',
                                      'Radius',
                                      'Neighbour',
                                      'Reconstruction_Mean',
                                      'Reconstruction_Var', 
                                      'NLN_Mean',
                                      'NLN_Var'])


    df_out = None 
    for f in glob('outputs/results_{}_*'.format(dataset)):
        if ('updated' in f): continue

        print('Loading data from {}'.format(f[16:-4]))
        if df_out is None:
            df_out =  add_df_parameters(f[16:-4])
        else: df_out = df_out.append(add_df_parameters(f[16:-4]))
    df_out = df_out[(df_out.Radius != 1) & (df_out.Radius != 2)]

    df_agg = df_out.groupby(['Model',
                             'Class',
                             'Latent_Dim',
                             'Radius',
                             'Neighbour']).agg({'AUC_Reconstruction_Error' :['mean','var'],
                             'AUC_Latent_Error' :['mean','var']}).reset_index()

    # Average across all classes to see which has highest average AUROC curve
    df_temp= df_out.groupby(['Model',
                             'Latent_Dim',
                             'Radius',
                             'Neighbour']).agg({'AUC_Reconstruction_Error' :['mean','var'],
                             'AUC_Latent_Error' :['mean','var']}).reset_index()

    max_indx = df_temp['AUC_Latent_Error']['mean'].idxmax()
    model = df_temp.iloc[max_indx].Model.values[0]
    r = float(df_temp.iloc[max_indx].Radius.values)
    K = int(df_temp.iloc[max_indx].Neighbour)
    L = int(df_temp.iloc[max_indx].Latent_Dim)


    df = df_agg[(df_agg.Model == model) &
                (df_agg.Radius == r) &
                (df_agg.Neighbour== K) &
                (df_agg.Latent_Dim== L) ]

    print('Model = {}, r = {}, K = {}, L = {}'.format(model,r,K,L))
    print('The best average performing model is')
    print(df) 

    return df_agg, df




def load_data(dataset):
    """
        load the specified dataset
    """
    if dataset == 'MNIST':
        data  = load_mnist()

    elif dataset == 'FASHION_MNIST':
        data  = load_fashion_mnist()

    elif dataset  == 'CIFAR10':
        data  = load_cifar10()

    (train_dataset,train_images,train_labels,test_images,test_labels) = data
    return train_images, train_labels, test_images, test_labels

