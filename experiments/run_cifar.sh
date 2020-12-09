#!/bin/sh
echo "Logging for run_cifar.sh at time: $(date)." >> log.log

limit=1000 #None 
epochs=25 
percentage=0.0
seed=$(openssl rand -hex 3)
d=$(date +'%m-%d-%Y-%I-%M_')

for i in $(seq 0 9)
	do
	for ld in 2 5 10 50 100
		do
				python -u main.py -limit $limit \
							      -anomaly_class $i \
							      -percentage_anomaly $percentage \
							      -epochs $epochs \
							      -latent_dim $ld \
							      -data CIFAR10\
							      -neighbors 1 2 4 5 10 100 \
							      -radius 1 2 5 10 20 100 \
							      -algorithm radius \
								  -seed $d$seed | tee -a cifar.log 
									
		done
done

python report.py -data CIFAR10 -seed $seed
