bash synch_servers.sh

export num_nodes=`wc -l < hosts.txt`
export num_workers_per_node=4  #TensorFlow instances per node
export num_processes=$(( $num_nodes * $num_workers_per_node ))
export ppr=2   # Two processes per resource (e.g. socket)
export physical_cores=`lscpu | grep "Core(s) per socket" | cut -d':' -f2 | sed "s/ //g"`
export pe=`lscpu | grep "Core(s) per socket" | cut -d':' -f2 | sed "s/ //g"` #$(( $physical_cores / $ppr ))
export num_threads=$(( $ppr * $physical_cores ))

mpirun -np $num_processes --hostfile hosts.txt -bind-to none \
        --map-by ppr:$ppr:socket:pe=$pe \
	--report-bindings --oversubscribe \
        bash hvd_multiworker.sh _multiworker $ppr $num_threads
