import numpy as np
import pandas as pd
from os import path
from matplotlib import pyplot as plt
from tensorflow.keras import Sequential
from sklearn.metrics import roc_curve, auc, f1_score
from models import Autoencoder, Encoder, Discriminator_x
from model_loader import get_error
from data import load_mnist


MODEL_PATH = '/tmp/models'
MODEL = 'AE'
LD =  10
DIM = (28,28,1)

class Namespace:
    """
        A hack to emulate input arguments
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def load_model(anomaly):
    """
        Load model
    """
    args = Namespace(latent_dim = LD,
                     input_shape=DIM)
    ae = Autoencoder(args)
    encoder = Sequential(Encoder(args))
    ae.load_weights(path.join(MODEL_PATH,
                              MODEL,
                              str(LD),
                              str(anomaly),
                              'training_checkpoints',
                              'checkpoint_full_model_ae'))

    return ae

def get_auroc(error ,labels, anomaly):
    """
        gets AUROC
    """

#    error = (error - np.min(error))/(np.max(error) - np.min(error))
    return roc_curve(labels==anomaly, error)


def get_errors_ae(model,test_images,test_labels,anomaly):
    """
        get_errors of model
    """
    error_ae = get_error('AE',
                         model,
                         test_images)

    fpr, tpr, thr = get_auroc(error_ae,
                                 test_labels,
                                 anomaly)
    _auc = auc(fpr,tpr)
    ind = np.argmax(tpr-fpr)
    print('AUC = {}, TPR = {}, FPR = {}'.format(_auc,
                                                tpr[ind],
                                                fpr[ind]))
    
    model_output = model(test_images)
    error = np.abs(test_images- model_output.numpy())

    return error,_auc,fpr,tpr,thr[ind],error_ae>thr[ind]

def plot(ax, test_images,test_labels, error, thr, inds,anom):
    """
        Generate plots for model
    """
    for i in range(10): # show all 10 digits
        ax[i].imshow(error[inds[i],...,0], vmin=0,vmax=1)
        ax[i].set_title(anom[inds[i]],fontsize=6,fontweight='bold')


def main():
    fig,axs = plt.subplots(10, 10)
    _,_,_,test_images,test_labels = load_mnist(anomaly=None, limit=None)
    inds = list(np.unique(test_labels, return_index =True)[1])

    for a,ax in zip(range(10),axs):
        print('Anomaly = {}'.format(a))
        #_np_inds  = np.where(test_labels==a)[0][:10]
        #inds = [int(i) for i in _np_inds]

        model  = load_model(anomaly=a)
        error, _auc, fpr, tpr, thr, anom = get_errors_ae(model,test_images, test_labels,a) 
        plot(ax,test_images,test_labels,error,thr, inds,anom)
#        ax[0].set_ylabel('AUCROC = {}\nTPR={}\nFPR={}'.format(round(_auc[ind],3),round(tpr[ind],3),round(fpr[ind],3)), fontsize=6) 
    plt.show()

if __name__ == '__main__':
    main()


def plot_AUCs(test_images,test_labels,anomaly, tpr,fpr,thr,error,error_bar,tpr_t = 0.7):
    fig,axs = plt.subplots(11,10)
    l = int(np.sqrt(len(test_labels)))
    plt.title('Error plots for anomaly = {}'.format(anomaly))
    axs[0,0].imshow(np.reshape(test_labels == anomaly,(l,l)))
   
    _thr  = thr[len(thr)-100:]# take the last 100 thresholds  
    for i in range(1,11):
        for j in range(10):
            ind = 10*(i-1) + j 
            e = error_bar>_thr[ind]
            axs[i,j].imshow(np.reshape(e,(l,l)))
