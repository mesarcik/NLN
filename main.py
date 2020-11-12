import numpy as np 
import tensorflow as tf
import sys
from data import *
from utils import cmd_input 
from reporting import plot_results
from architectures import *  

def main():
    elif cmd_input.args.data == 'MNIST':
        data  = load_mnist(limit=cmd_input.args.limit,
                           anomaly=cmd_input.args.anomaly_class,
                           percentage_anomaly =cmd_input.args.percentage_anomaly)

    elif cmd_input.args.data == 'FASHION_MNIST':
        data  = load_fashion_mnist(limit=cmd_input.args.limit,
                           anomaly=cmd_input.args.anomaly_class,
                           percentage_anomaly =cmd_input.args.percentage_anomaly)

    elif cmd_input.args.data == 'CIFAR10':
        data  = load_cifar10(limit=cmd_input.args.limit,
                           anomaly=cmd_input.args.anomaly_class,
                           percentage_anomaly =cmd_input.args.percentage_anomaly)

    (train_dataset,train_images,train_labels,test_images,test_labels) = data

    print(" __________________________________ \n Anomaly class {}".format(
                                               cmd_input.args.anomaly_class))
    print(" __________________________________ \n Latent dimensionality {}".format(
                                               cmd_input.args.latent_dim))
    print(" __________________________________ \n Save name {}".format(
                                               cmd_input.args.model_name))
    print(" __________________________________ \n")

    train_dae(train_dataset,train_images,train_labels,test_images,test_labels,cmd_input.args)
    train_ganomaly(train_dataset,train_images,train_labels,test_images,test_labels,cmd_input.args)
    train_ae(train_dataset,train_images,train_labels,test_images,test_labels,cmd_input.args)
    train_vae(train_dataset,train_images,train_labels,test_images,test_labels,cmd_input.args)
    train_aae(train_dataset,train_images,train_labels,test_images,test_labels,cmd_input.args)

    #train_gpnd(train_dataset,train_images,train_labels,test_images,test_labels,cmd_input.args)
    #train_vaegan(train_dataset,train_images,train_labels,test_images,test_labels,cmd_input.args)
    #train_bigan(train_dataset,train_images,train_labels,test_images,test_labels,cmd_input.args)

if __name__ == '__main__':
    main()
