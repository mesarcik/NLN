#!/bin/sh
echo "Logging for run_mvtecr.sh at time: $(date)." >> log.log

limit=1000 #None 
epochs=50
percentage=0.0
seed=$(openssl rand -hex 3)
d=$(date +'%m-%d-%Y-%I-%M_')

for i in bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper 
	do
	for ld in 2 5 10 50 100
		do
				python -u main.py -limit $limit \
							      -anomaly_class $i\
							      -percentage_anomaly $percentage \
							      -epochs $epochs \
							      -latent_dim $ld \
							      -data MVTEC\
							      -neighbors 1 2 4 5 10 100 \
							      -radius 1 2 5 10 20 100 \
							      -algorithm radius \
								  -seed $d$seed | tee -a mvtec.log 
		done
done

python report.py -data MVTEC -seed $seed
