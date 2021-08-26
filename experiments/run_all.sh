#!/bin/sh
echo "Logging for run_mnist.sh at time: $(date)." >> log.log

limit=None
epochs=0
percentage=0.0
seed=$(openssl rand -hex 3)
d=$(date +'%m-%d-%Y-%I-%M_')
ld=2048

for data in CIFAR10 MNIST FASHION_MNIST  
do
		for atype in SIMO MISO 
		do
				for i in $(seq 0 9)
				do
						python -u main.py -limit $limit \
										  -anomaly_type $atype \
										  -anomaly_class $i \
										  -percentage_anomaly $percentage \
										  -epochs $epochs \
										  -latent_dim $ld \
										  -data $data \
										  -neighbors 1 3 2 5 10\
										  -algorithm knn\
										  -seed $d$seed | tee -a $data.log 
				done
		done
done 
