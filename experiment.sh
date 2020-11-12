#!/bin/sh

# Note -u is used to get an unbuffered output from python
# and unbuffer -p tee is used to get an unbuffered output to the screen and file
echo "Logging for experiment.sh on Time: $(date)." >> log.log

limit=None #16384
epochs=25 #<- note i am training for more epochs now. It was previously 15 
j=0.0

for i in 0 1 2 3 4 5 6 7 8 9
	do
	for ld in 2 5 10 50 100
		do
				python -u main.py -limit $limit \
							      -anomaly_class $i \
							      -percentage_anomaly $j \
							      -epochs $epochs \
							      -latent_dim $ld \
							      -data MNIST\
							      -neighbors 1 2 4 5 10 100 \
							      -radius 1 2 5 10 20 100 \
							      -algorithm radius | unbuffer -p tee -a log.log 
		done
done
#for i in 1 2 3 4 5 6 7 8 9
#	do
#	for ld in 2 5 10 50 100
#		do
#				python -u main.py -limit $limit \
#							      -anomaly_class $i \
#							      -percentage_anomaly $j \
#							      -epochs $epochs \
#							      -latent_dim $ld \
#							      -data FASHION_MNIST\
#							      -neighbors 1 2 4 5 10 100 \
#							      -radius 1 2 5 10 20 100 \
#							      -algorithm radius | unbuffer -p tee -a log.log 
#		done
#done


#for i in point_source diffuse_foreground rfi_scatter rfi_dtv rfi_impulse gains rfi_stations x_talk
