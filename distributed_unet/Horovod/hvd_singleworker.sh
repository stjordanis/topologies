source activate tf
export OMP_NUM_THREADS=${3}
python hvd_train.py --epochs=15 --learningrate=0.0005 --logdir="tensorboard${1}" --num_inter_threads=${2} --num_threads=${3}
source deactivate
