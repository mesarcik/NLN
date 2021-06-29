#!/bin/sh
echo "Logging for run_mnist.sh at time: $(date)." >> log.log

limit=None
epochs=50
percentage=0.0
seed=$(openssl rand -hex 3)
d=$(date +'%m-%d-%Y-%I-%M_')

for data in MNIST FASHION_MNIST CIFAR10 
do
		for ld in 8 16 32 64 
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
												  -latent_dim $ld \
												  -data $data \
												  -neighbors 1 3 2 5 10\
												  -algorithm knn\
												  -seed $d$seed | tee -a $data.log 
						done
				done
		done
done 
