#!/bin/sh
echo "Logging for run_mvtecr.sh at time: $(date)." >> log.log

limit=None
epochs=-1
percentage=0.0
seed=$(openssl rand -hex 3)
d=$(date +'%m-%d-%Y-%I-%M_')
ld=2048

for patch in 128 
	do
		for i in bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper
		do
				python -u main.py -limit $limit \
							      -anomaly_class $i\
							      -percentage_anomaly $percentage \
							      -epochs $epochs \
							      -latent_dim $ld\
							      -data MVTEC\
							      -neighbors 2 \
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
