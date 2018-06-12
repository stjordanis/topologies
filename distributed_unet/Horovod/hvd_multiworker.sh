source activate tf
export OMP_NUM_THREADS=lscpu | grep "Core(s) per socket" | cut -d':' -f2 | sed "s/ //g"
python hvd_train.py --epochs=15 --logdir="tensorboard${1}"
source deactivate
