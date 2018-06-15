logdir=${1:-_singleworker}
bash synch_servers.sh

export num_nodes=3
export inter_op=2
export physical_cores=48

mpirun -np $num_nodes -H 192.168.5.152:1,192.168.5.153:1,192.168.5.154:1 -map-by slot -bind-to none bash hvd_singleworker.sh $logdir $inter_op $physical_cores
