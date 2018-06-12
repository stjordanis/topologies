bash synch_servers.sh
mpirun -np 3 -H 192.168.5.152:1,192.168.5.153:1,192.168.5.154:1 -bind-to none -map-by slot bash hvd_singleworker.sh _singleworker
