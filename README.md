# `NLN`: Nearest-Latent-Neighbours
A repository containing the implementation of the paper entitled "Nearest Neighbours Improves Novelty Detection of Autoencoders"

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

Additionally for training on a GPU run:
``` 
    conda install -c anaconda tensorflow-gpu=2.1.0
``` 


## Replication of results in paper 
Run the following to replicate the results for MNIST, CIFAR-10, Fashion-MNIST, MVTec-AD respectively
```
    sh experiments/run_mnist.sh
    sh experiments/run_cifar.sh
    sh experiments/run_fmnist.sh
    sh experiments/run_mvtec.sh
```

Or to execute all experiments sequentially the following script can be run:
```
    sh experiments/run_all.sh
```

### MVTec-AD usage 
You will need to download the [MVTec anomaly detection dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad) and specify the its path using `-mvtec_path` command line option.

## Training 
Run the following: 
```
    python main.py -anomaly_class <0,1,2,3,4,5,6,7,8,9> \
                   -percentage_anomaly <float> \
                   -limit <int> \
                   -epochs <int> \
                   -latent_dim <int> \
                   -data <MNIST,FASHION_MNIST,CIFAR10,MVTEC> \
                   -mvtec_path <str>\
                   -neighbors <int(s)> \
                   -algorithm <knn> \
		   -patches <True, False> \
		   -crop <True, False> \
		   -rotate <True, False> \
		   -patch_x <int> \    
		   -patch_y <int> \    
		   -patch_x_stride <int> \    
		   -patch_y_stride <int> \    
		   -crop_x <int> \    
		   -crop_y <int> \    
```
## Reporting Results 
Run the following given the correctly generated results files:
```
    python report.py -data <MNIST,CIFAR10,FASHION_MNIST,MVTEC> -seed <filepath-seed>
```

## Licensing
Source code of NLN is licensed under the MIT License.
