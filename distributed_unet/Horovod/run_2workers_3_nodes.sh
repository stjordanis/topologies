#!/bin/bash
echo "Synching hosts.."
bash synch_servers.sh

for i in {1..10}
do

   HOROVOD_FUSION_THRESHOLD=134217728 /usr/local/openmpi/bin/mpirun \
        -x HOROVOD_FUSION_THRESHOLD \
	-x OMP_NUM_THREADS=24 \
        -np 6 -H 192.168.5.152,192.168.5.153,192.168.5.154  \
        --map-by socket -cpus-per-proc 24 \
	--report-bindings --oversubscribe \
        bash /home/genodeuser1/topologies/distributed_unet/Horovod/test.sh 48 2 0.0005 tensorboard_4worker_3nodes_run${i} \
        2>&1 | tee train_unet_4_worker_3nodes_run${i}.log 

   bash clear_caches.sh

done



