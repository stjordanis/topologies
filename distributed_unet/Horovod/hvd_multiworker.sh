source activate tf
export OMP_NUM_THREADS=${3}
python hvd_train.py --learningrate=0.00025 --logdir="tensorboard${1}" --num_inter_threads=${2} --num_threads=${3}
source deactivate
