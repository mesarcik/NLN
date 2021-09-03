#!/bin/sh
echo "Logging for run_mvtec.sh at time: $(date)." >> log.log

limit=None
epochs=50
percentage=0.0
seed=$(openssl rand -hex 3)
d=$(date +'%m-%d-%Y-%I-%M_')
ld=128
atype=SIMO

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
							      -mvtec_path /home/mmesarcik/data/MVTecAD/\
							      -neighbors 2 5 \
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

python report.py -data MVTEC\ 
                 -seed $seed\
		 -anomaly_type $atype\
		 -mvtec_path datasets/MVTecAD/\
		 -anomaly_class nan\
