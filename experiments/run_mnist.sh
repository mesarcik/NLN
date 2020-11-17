#!/bin/sh
echo "Logging for run_mnist.sh at time: $(date)." >> log.log

limit=None 
epochs=25 
percentage=0.0

for i in $(seq 0 9)
	do
	for ld in 2 5 10 50 100
		do
				python -u main.py -limit $limit \
							      -anomaly_class $i \
							      -percentage_anomaly $percentage \
							      -epochs $epochs \
							      -latent_dim $ld \
							      -data MNIST\
							      -neighbors 1 2 4 5 10 100 \
							      -radius 1 2 5 10 20 100 \
							      -algorithm radius | tee -a mnist.log 
		done
done

python report.py -data MNIST
