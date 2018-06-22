#!/bin/bash
echo "Synching hosts.."
bash synch_servers.sh

for i in {2..10}
do

   HOROVOD_FUSION_THRESHOLD=134217728 /usr/local/openmpi/bin/mpirun \
        -x HOROVOD_FUSION_THRESHOLD \
	-x OMP_NUM_THREADS=24 \
        -np 6 -H 192.168.5.152,192.168.5.153,192.168.5.154  \
        --map-by socket -cpus-per-proc 24 \
	--report-bindings --oversubscribe \
        bash /home/genodeuser1/topologies/distributed_unet/Horovod/test.sh 48 2 0.0001 tensorboard_1worker_3nodes_run${i}

   bash clear_caches.sh

done



