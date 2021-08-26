#!/bin/sh
echo "Logging for run_cifar.sh at time: $(date)." >> log.log

limit=None
epochs=-1
percentage=0.0
seed=$(openssl rand -hex 3)
d=$(date +'%m-%d-%Y-%I-%M_')
ld=2048

for atype in SIMO #MISO 
do
		for i in $(seq 0 9)
		do
				python -u main.py -limit $limit \
								  -anomaly_class $i \
								  -anomaly_type $atype \
								  -percentage_anomaly $percentage \
								  -epochs $epochs \
								  -latent_dim $ld \
								  -data CIFAR10\
								  -neighbors 1 2 5 10 \
								  -algorithm knn\
								  -seed $d$seed | tee -a cifar.log 
	   done
done

