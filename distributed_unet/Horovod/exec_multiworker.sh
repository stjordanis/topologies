# Execute training on each node, pulling local thread counts for the train script

source activate tf
export physical_cores=`lscpu | grep "Core(s) per socket" | cut -d':' -f2 | sed "s/ //g"` # Total number of physical cores per socket
export num_threads=$(( ${2} * $physical_cores )) # Total number of physical cores on this machine

#python hvd_train.py --learningrate=0.00025 --logdir="tensorboard${1}" --num_inter_threads=${3} --num_threads=$num_threads
python hvd_train.py --num_warmups=1 --learningrate=0.0006 --logdir="tensorboard${1}" --num_inter_threads=${3} --num_threads=$num_threads
source deactivate
