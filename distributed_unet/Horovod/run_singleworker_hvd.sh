logdir=${1:-_singleworker}     # Default suffix is _singeworker
node_ips=${2:-hosts.txt}      # Default is the hosts.txt file
export num_workers_per_node=${3:-1}  # Default 1 worker per node

export num_nodes=`wc -l < ${node_ips}`
export num_processes=$(( $num_nodes * $num_workers_per_node ))
export ppr=2   # Two processes per resource (e.g. socket)
export physical_cores=`lscpu | grep "Core(s) per socket" | cut -d':' -f2 | sed "s/ //g"`
export pe=`lscpu | grep "Core(s) per socket" | cut -d':' -f2 | sed "s/ //g"` #$(( $physical_cores / $ppr ))
export num_threads=$(( $ppr * $physical_cores ))

bash synch_servers.sh

mpirun -np $num_processes --hostfile $node_ips -bind-to none \
        --map-by ppr:$ppr:socket:pe=$pe \
	--report-bindings --oversubscribe \
        bash hvd_singleworker.sh $logdir $ppr $num_threads
