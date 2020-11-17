# `NLN`: Nearest-Latent-Neighbours
A repository containing the implementation of the paper entitled "Nearest Neighbours Improves Multi-class Novelty Detection of Autoencoders"

## Installation 
Install conda environment by:
``` 
    conda create --name nln python=3.7
``` 
Run conda environment by:
``` 
    conda activate nln
``` 

Install dependancies by running:
``` 
    pip install -r dependancies
``` 

## Replication of results in paper 
Run the following to replicate the results for MNIST, CIFAR-10 and Fashion-MNIST respectively
```
    sh experiments/run_mnist.sh
    sh experiments/run_cifar.sh
    sh experiments/run_fmnist.sh
```


## Training 
Run the following given the correctly generated training files
```
    python main.py -anomaly_class <0,1,2,3,4,5,6,7,8,9> \
                   -percentage_anomaly <float> \
                   -limit <int> \
                   -epochs <int> \
                   -latent_dim <int> \
                   -data <MNIST,FASHION_MNIST,CIFAR10> \
                   -neighbors <int(s)> \
                   -radius <float(s)> \
                   -algorithm <knn,radius>    
```

## Licensing
Source code of NLN is licensed under the MIT License.
