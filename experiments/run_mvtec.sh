#!/bin/sh
echo "Logging for run_mvtecr.sh at time: $(date)." >> log.log

limit=None
epochs=200
percentage=0.0
seed=$(openssl rand -hex 3)
d=$(date +'%m-%d-%Y-%I-%M_')

for patch in 128 #64 32
	do
		for i in bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper
		do
				python -u main.py -limit $limit \
							      -anomaly_class $i\
							      -percentage_anomaly $percentage \
							      -epochs $epochs \
							      -latent_dim 128 \
							      -data MVTEC\
							      -neighbors 1 2 3 5 \
							      -radius 1 2 5 100 \
							      -algorithm knn\
								  -rotate True \
								  -crop True \
								  -patches True \
								  -crop_x $patch\
								  -crop_y $patch\
								  -patch_x $patch \
								  -patch_y $patch \
								  -patch_stride_x $patch \
								  -patch_stride_y $patch \
							      -seed $d$seed | tee -a mvtec.log
		done
done
#$d$seed 
#python report.py -data MVTEC -seed $seed
