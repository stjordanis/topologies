source activate tf
export OMP_NUM_THREADS=${3}
python hvd_train.py --logdir="tensorboard${1}" --num_inter_threads=${2} --num_threads=${3}
source deactivate
