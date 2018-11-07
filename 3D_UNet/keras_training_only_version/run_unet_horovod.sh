#!/usr/bin/env bash
source ~/.bashrc
conda activate tf
cd ~/topologies/3D_UNet/keras_training_only_version/
#python train.py --bz 8  --intraop_threads 26
python ~/topologies/3D_UNet/keras_training_only_version/keras_mnist.py
conda deactivate
