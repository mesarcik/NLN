#!/bin/sh
echo "Logging for run_mnist.sh at time: $(date)." >> log.log

limit=None
epochs=50
percentage=0.0
seed=$(openssl rand -hex 3)
d=$(date +'%m-%d-%Y-%I-%M_')

ld=64
for repeat in $(seq 0 1)
do
		for data in MNIST CIFAR10 FASHION_MNIST
		do
				for atype in MISO SIMO
				do
						for i in $(seq 0 9)
						do
								python -u main.py -limit $limit \
												  -anomaly_type $atype \
												  -anomaly_class $i \
												  -percentage_anomaly $percentage \
												  -epochs $epochs \
												  -latent_dim 32 \
												  -data $data \
												  -neighbors 1 3 2 5 10\
												  -algorithm knn\
												  -seed 06-03-2021-04-06_288a27| tee -a $data.log 
						done
				done
		done
done 
#-seed  $d$seed 
#python report.py -data MNIST -seed $d$seed
