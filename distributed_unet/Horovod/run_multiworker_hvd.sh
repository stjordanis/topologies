#!/bin/bash
# To run: bash run_multiworker_hvd.sh <logidr> <hostfile> <workers per node> <inter op threads>
# Note: The total number of workers deployed will be the number of workers per node * number of nodes

logdir=${1:-_multiworker}     # Default suffix is _multiworker
node_ips=${2:-hosts.txt}      # Default is the hosts.txt file
export num_workers_per_node=${3:-4}  # Default 4 workers per node
export num_inter_threads=${4:-2} # Default to 2 inter_op threads

export physical_cores=`lscpu | grep "Core(s) per socket" | cut -d':' -f2 | sed "s/ //g"` # Total number of physical cores per socket
export num_nodes=`cat ${node_ips} | sed '/^\s*$/d' | wc -l` # Hosts.txt should contain a single host per line
export num_processes=$(( $num_nodes * $num_workers_per_node )) # Total number of workers across all nodes
export ppr=2   # Number of sockets per node

echo "Running $num_workers_per_node worker(s)/node on $num_nodes nodes..."

echo "Synching hosts.."
bash synch_servers.sh

mpirun -np $num_processes --hostfile $node_ips -bind-to none \
        --map-by ppr:$ppr:socket:pe=$physical_cores \
	--report-bindings --oversubscribe \
        bash exec_multiworker.sh $logdir $ppr $num_inter_threads
