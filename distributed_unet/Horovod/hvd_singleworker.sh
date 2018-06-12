source activate tf
python hvd_train.py --epochs=15 --learningrate=0.0005 --logdir="tensorboard${1}"
source deactivate
