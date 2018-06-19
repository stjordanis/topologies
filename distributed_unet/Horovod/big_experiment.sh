for i in {1..10}; 
do

    python nohvd_train.py --logdir="tensorboard_no_horovod_${i}"
    bash clear_caches.sh

    #./run_multiworker_hvd.sh _4workers_multinode_run$i hosts.txt 4
    #bash clear_caches.sh

    ./run_singleworker_hvd.sh _1worker_singlenode_run$i localhost.txt 1
    bash clear_caches.sh

    ./run_singleworker_hvd.sh _4worker_singlenode_run$i localhost.txt 4
    bash clear_caches.sh

    ./run_singleworker_hvd.sh _1worker_multinode_run$i hosts.txt 1
    bash clear_caches.sh
    
done



