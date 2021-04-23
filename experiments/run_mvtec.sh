#!/bin/sh
echo "Logging for run_mvtecr.sh at time: $(date)." >> log.log

limit=None
epochs=200
percentage=0.0
seed=$(openssl rand -hex 3)
d=$(date +'%m-%d-%Y-%I-%M_')

for i in bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper
	do
	for ld in 64 128 1024
		do
				python -u main.py -limit $limit \
							      -anomaly_class $i\
							      -percentage_anomaly $percentage \
							      -epochs $epochs \
							      -latent_dim $ld \
							      -data MVTEC\
							      -neighbors 1 3 5 10 \
							      -radius 1 2 5 100 \
							      -algorithm knn\
								  -rotate True \
								  -crop True \
								  -crop_x 256\
								  -crop_y 256\
								  -patches True \
								  -patch_x 256 \
								  -patch_y 256 \
								  -patch_stride_x 256\
								  -patch_stride_y 256\
								  -seed $d$seed | tee -a mvtec.log 
		done
done

#python report.py -data MVTEC -seed $seed
