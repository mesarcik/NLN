import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from glob import glob

def add_df_parameters(args):
    """
        Adds the missing parameters to the results.csv file 

    """
    df = pd.read_csv('outputs/results_{}_{}.csv'.format(args.data, args.seed))
    df.drop(columns =list(df.columns[-7:]),inplace =True) 
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
                    d.Patch_Size, d.Class, d.Type, vals[0], 
                    d.AUC_Reconstruction_Error, vals[2], vals[3], vals[4], vals[5]]
            data.append(temp)

    df_new = pd.DataFrame(data,columns=df.columns)

    return df_new

def generate_tables(args,verbose=False):
    """
        Creates a single barplot for a given dataset 

    """
    TYPE= args.anomaly_type 
    ERROR = ['Distance_AUC','AUC_NLN_Error','Sum_Recon_NLN_Dist', 'Mul_Recon_NLN_Dist'][3]

    df = add_df_parameters(args)
    df = df[df.Type==TYPE]

    df_agg_recon = df.groupby(['Model',
                              'Class',
                              'Latent_Dim']).agg({'AUC_Reconstruction_Error': 'mean'}).reset_index()

    df_agg_nln = df.groupby(['Model',
                            'Class',
                            'Latent_Dim',
                            'Neighbour']).agg({ERROR: 'mean'}).reset_index()
    improvement = [] 
    performance, model_type, nneighbours, score,dim  = {}, '', -1, 0,0
    for model in ['AE','AAE', 'VAE', 'DAE_disc', 'GANomaly']: 
        recon,latent = [],[]
        for cl in np.sort(pd.unique(df.Class)):

            idx_latent = (df_agg_nln[(df_agg_nln.Model == model)][ERROR].idxmax())
            idx_recon = (df_agg_recon[(df_agg_recon.Model == model)].AUC_Reconstruction_Error.idxmax())

            ld = df_agg_nln.iloc[idx_latent]['Latent_Dim']
            neigh  = df_agg_nln.iloc[idx_latent]['Neighbour']

            df_max = df_agg_nln[(df_agg_nln.Model == model) &
                                (df_agg_nln.Latent_Dim == ld) &
                                (df_agg_nln.Neighbour == neigh) &
                                (df_agg_nln.Class == cl)]

            latent.append(df_max[ERROR].values[0])

            ld = df_agg_recon.iloc[idx_recon]['Latent_Dim']

            df_max = df_agg_recon[(df_agg_recon.Model == model) &
                        (df_agg_recon.Latent_Dim == ld) &
                        (df_agg_recon.Class == cl)]

            recon.append(df_max['AUC_Reconstruction_Error'].values[0])

        improvement.append(round(100*(1-np.mean(recon)/np.mean(latent)),2))
        performance[model] = [neigh, np.mean(latent),ld]

        if np.mean(latent) > score: 
            score = np.mean(latent)
            model_type = model
            nneighbours = neigh
            dim = ld

        if verbose:
            print('Model {} \nMean Reconstruction error {}+-{}\nMean Latent Error {}+-{}\nPercetange Increase {}\n'.format(model,
                                                                                                                            round(np.mean(recon),3), 
                                                                                                                            round(np.var(recon),2), 
                                                                                                                            round(np.mean(latent),3), 
                                                                                                                            round(np.var(latent),2), 
                                                                                                                            round(1-np.mean(recon)/np.mean(latent),4)))
    #print('\t & {}\%\t & {}\%\t & {}\%\t & {}\%\t & {}\% \\'.format('AE','AAE', 'VAE', 'DAE', 'GANomaly'))
    print('Dataset: {} \t Model ={} \t #Neigbours={} \t Latent Dim={}  \t nAUCROC={}'.format(args.data, model_type, nneighbours, dim, round(score,3)))

    if args.data== 'FASHION_MNIST': d = 'FMNIST'
    else: d = args.data
    print('{}\t & {}\%\t & {}\%\t & {}\%\t & {}\%\t & {}\%\t\\'.format(d,*improvement)) # & {}\%\t & {}\%\t & {}\%\t & {}\%
