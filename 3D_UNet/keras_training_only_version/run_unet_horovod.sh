#!/usr/bin/env bash
source activate tf
python /home/bduser/topologies/3D_UNet/keras_training_only_version/train.py --horovod
source deactivate


