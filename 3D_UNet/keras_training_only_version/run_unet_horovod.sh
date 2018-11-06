#!/usr/bin/env bash
source ~/.bashrc
conda activate tf
python ~/topologies/3D_UNet/keras_training_only_version/train.py --bz 8 
#python ~/topologies/3D_UNet/keras_training_only_version/keras_mnist.py
conda deactivate
