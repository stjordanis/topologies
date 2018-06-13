bash synch_servers.sh
export physical_cores=`lscpu | grep "Core(s) per socket" | cut -d':' -f2 | sed "s/ //g"`
mpirun -np 12 --hostfile hosts.txt -bind-to none --map-by ppr:2:socket:pe=$physical_cores bash hvd_multiworker.sh _multiworker
