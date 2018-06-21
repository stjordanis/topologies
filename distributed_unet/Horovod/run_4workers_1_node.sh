#!/bin/bash
echo "Synching hosts.."
bash synch_servers.sh

HOROVOD_FUSION_THRESHOLD=134217728 /usr/local/openmpi/bin/mpirun \
        -x HOROVOD_FUSION_THRESHOLD \
	-x OMP_NUM_THREADS=12 \
        -np 4 -H localhost  \
        --map-by socket -cpus-per-proc 12 \
	--report-bindings --oversubscribe \
        bash /home/genodeuser1/topologies/distributed_unet/Horovod/test.sh 12 2 0.00025


