#!/usr/bin/env bash
source ~/.bashrc
conda activate tf
cd ~/topologies/3D_UNet/keras_training_only_version/
python train_horovod.py --bz 8  --intraop_threads 28
#python ~/topologies/3D_UNet/keras_training_only_version/keras_mnist.py
conda deactivate
