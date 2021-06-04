import numpy as np 
import tensorflow as tf
import sys
from data import *
from utils import cmd_input 
from architectures import *  

def main():
    """
        Reads data and cmd arguments and trains models
    """
    if cmd_input.args.data == 'MNIST':
        data  = load_mnist(cmd_input.args)

    elif cmd_input.args.data == 'FASHION_MNIST':
        data  = load_fashion_mnist(cmd_input.args)

    elif cmd_input.args.data == 'CIFAR10':
        data  = load_cifar10(cmd_input.args)

    if cmd_input.args.data == 'MVTEC':
        data  = load_mvtec(cmd_input.args)
        test_masks = data[5]

    else: test_masks = None

    (train_dataset,train_images,train_labels,test_images,test_labels) = data[0:5]


    print(" __________________________________ \n Anomaly class {}".format(
                                               cmd_input.args.anomaly_class))
    print(" __________________________________ \n Latent dimensionality {}".format(
                                               cmd_input.args.latent_dim))
    print(" __________________________________ \n Save name {}".format(
                                               cmd_input.args.model_name))
    print(" __________________________________ \n")

    with open("temp_log", "a") as f:
        f.write('{} \t {}\n'.format(cmd_input.args.anomaly_class, cmd_input.args.model_name))

    #train_ae_resnet(train_dataset,train_images,train_labels,test_images,test_labels, test_masks, cmd_input.args)
    #train_nnae(train_dataset,train_images,train_labels,test_images,test_labels, test_masks, cmd_input.args)
    #train_ae_ssim(train_dataset,train_images,train_labels,test_images,test_labels, test_masks, cmd_input.args)
    train_ae(train_dataset,train_images,train_labels,test_images,test_labels, test_masks, cmd_input.args)
    train_dae(train_dataset,train_images,train_labels,test_images,test_labels, test_masks, cmd_input.args)
    train_ganomaly(train_dataset,train_images,train_labels,test_images,test_labels,test_masks, cmd_input.args)
    train_vae(train_dataset,train_images,train_labels,test_images,test_labels, test_masks, cmd_input.args)
    train_aae(train_dataset,train_images,train_labels,test_images,test_labels, test_masks, cmd_input.args)



if __name__ == '__main__':
    main()
