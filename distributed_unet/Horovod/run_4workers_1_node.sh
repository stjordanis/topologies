#!/bin/bash
echo "Synching hosts.."
bash synch_servers.sh

for i in {3..10}
do

     echo "Run #${i}"
     HOROVOD_FUSION_THRESHOLD=134217728 /usr/local/openmpi/bin/mpirun \
        -x HOROVOD_FUSION_THRESHOLD \
	-x OMP_NUM_THREADS=12 \
        -np 4 -H localhost  \
        --map-by socket -cpus-per-proc 12 \
	--report-bindings --oversubscribe \
        bash /home/genodeuser1/topologies/distributed_unet/Horovod/test.sh 12 2 0.00025 tensorboard_4workers_1_node_run${i}

     bash /home/genodeuser1/topologies/distributed_unet/Horovod/clear_caches.sh
done

