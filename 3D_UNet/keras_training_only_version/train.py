#!/usr/bin/python

# ----------------------------------------------------------------------------
# Copyright 2018 Intel
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

import numpy as np

import random
import os
import argparse
import psutil
import time
import datetime
import tensorflow as tf
from model import *

from dataloader import DataGenerator

parser = argparse.ArgumentParser(
    description="Train 3D U-Net model", add_help=True)
parser.add_argument("--bz",
                    type=int,
                    default=8,
                    help="Batch size")
parser.add_argument("--patch_dim",
                    type=int,
                    default=128,
                    help="Size of the 3D patch")
parser.add_argument("--lr",
                    type=float,
                    default=0.001,
                    help="Learning rate")
parser.add_argument("--train_test_split",
                    type=float,
                    default=0.85,
                    help="Train test split (0-1)")
parser.add_argument("--epochs",
                    type=int,
                    default=30,
                    help="Number of epochs")
parser.add_argument("--intraop_threads",
                    type=int,
                    default=psutil.cpu_count(logical=False)-4,
                    help="Number of intraop threads")
parser.add_argument("--interop_threads",
                    type=int,
                    default=2,
                    help="Number of interop threads")
parser.add_argument("--blocktime",
                    type=int,
                    default=0,
                    help="Block time for CPU threads")
parser.add_argument("--print_model",
                    action="store_true",
                    default=False,
                    help="Print the summary of the model layers")
parser.add_argument("--use_upsampling",
                    action="store_true",
                    default=False,
                    help="Use upsampling instead of transposed convolution")
datapath="/home/bduser/data/Brats2018/MICCAI_BraTS_2018_Data_Training"
parser.add_argument("--data_path",
                    default=datapath,
                    help="Root directory for BraTS 2018 dataset")
parser.add_argument("--saved_model",
                    default="./saved_model/3d_unet_brats2018.hdf5",
                    help="Save model to this path")
parser.add_argument("--horovod",
                    action="store_true",
                    default=False,
                    help="Use Horovod")

args = parser.parse_args()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Get rid of the AVX, SSE warnings
os.environ["OMP_NUM_THREADS"] = str(args.intraop_threads)
os.environ["KMP_BLOCKTIME"] = str(args.blocktime)
os.environ["KMP_AFFINITY"] = "granularity=thread,compact,1,0"

if args.horovod:
    import horovod.keras as hvd
    hvd.init()

    if hvd.rank()==0:
        os.system("lscpu")
        print("Started script on {}".format(datetime.datetime.now()))

        print("args = {}".format(args))
        os.system("uname -a")
        print("TensorFlow version: {}".format(tf.__version__))

        print("Keras API version: {}".format(K.__version__))
else:
    os.system("lscpu")
    print("Started script on {}".format(datetime.datetime.now()))

    print("args = {}".format(args))
    os.system("uname -a")
    print("TensorFlow version: {}".format(tf.__version__))

    print("Keras API version: {}".format(K.__version__))


# Optimize CPU threads for TensorFlow
config = tf.ConfigProto(
    inter_op_parallelism_threads=args.interop_threads,
    intra_op_parallelism_threads=args.intraop_threads)

sess = tf.Session(config=config)

import keras as K

K.backend.set_session(sess)


def get_file_list(data_path=args.data_path):
    """
    Get list of the files from the BraTS raw data
    Split into training and testing sets.
    """
    fileList = []
    for subdir, dir, files in os.walk(data_path):
        # Make sure directory has data
        if os.path.isfile(os.path.join(subdir,
                                       os.path.basename(subdir)
                                       + "_flair.nii.gz")):
            fileList.append(subdir)

    random.Random(816).shuffle(fileList)
    n_files = len(fileList)

    train_length = int(args.train_test_split*n_files)
    trainList = fileList[:train_length]
    testList = fileList[train_length:]

    return trainList, testList

input_shape = [args.patch_dim, args.patch_dim, args.patch_dim, 1]

if args.horovod:
    if (hvd.rank()==0):
        print_summary=args.print_model
    else:
        print_summary=args.print_model
else:
    print_summary=args.print_model

model = unet_3d(input_shape=input_shape,
                use_upsampling=args.use_upsampling,
                learning_rate=args.lr,
                n_cl_out=1,  # single channel (greyscale)
                dropout=0.5,
                print_summary=print_summary, using_horovod=args.horovod)


start_time = time.time()

# Save best model to hdf5 file
directory = os.path.dirname(args.saved_model)
try:
    os.stat(directory)
except:
    os.mkdir(directory)

# if os.path.isfile(args.saved_model):
#     model.load_weights(args.saved_model)

checkpoint = K.callbacks.ModelCheckpoint(args.saved_model,
                                         verbose=1,
                                         save_best_only=True)

# TensorBoard
tb_logs = K.callbacks.TensorBoard(log_dir="./tensorboard_logs")

if args.horovod:

    hvd_callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),

    # Horovod: average metrics among workers at the end of every epoch.
    #
    # Note: This callback must be in the list before the ReduceLROnPlateau,
    # TensorBoard or other metrics-based callbacks.
    hvd.callbacks.MetricAverageCallback(),

    # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
    # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
    # the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
    hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=1)]

    callbacks_list = hvd_callbacks + [checkpoint, tb_logs]

else:
    callbacks_list = [checkpoint, tb_logs]

# Separate file lists into train and test sets
trainList, testList = get_file_list()
with open("trainlist.txt", "w") as f:
    for item in trainList:
        f.write("{}\n".format(item))

with open("testlist.txt", "w") as f:
    for item in testList:
        f.write("{}\n".format(item))

if args.horovod:
    if hvd.rank()==0:
        print("Number of training MRIs = {}".format(len(trainList)))
        print("Number of test MRIs = {}".format(len(testList)))
else:
    print("Number of training MRIs = {}".format(len(trainList)))
    print("Number of test MRIs = {}".format(len(testList)))

# Run the script  "load_brats_images.py" to generate these Numpy data files
imgs_test = np.load(os.path(sys.path[0],"imgs_test_3d.npy"))
msks_test = np.load(os.path(sys.path[0],"msks_test_3d.npy"))

training_data_params = {"dim": (args.patch_dim,args.patch_dim,args.patch_dim),
               "batch_size": args.bz,
               "n_in_channels": 1,
               "n_out_channels": 1,
               "augment": True,
               "shuffle": True}

if args.horovod:
    shardLen = len(trainList) // hvd.size()
    startShard = hvd.rank()*shardLen
    if (hvd.rank() + 1) == hvd.size():
        stopShard = len(trainList)
    else:
        stopShard = (hvd.rank()+1)*shardLen

    shard = trainList[slice(startShard, stopShard)]
    training_generator = DataGenerator(shard, **training_data_params)
else:
    training_generator = DataGenerator(trainList, **training_data_params)

# validation_data_params = {"dim": (args.patch_dim,args.patch_dim,args.patch_dim),
#                "batch_size": 1,  # Use 1 so that we don't have partial batch
#                "n_in_channels": 1,
#                "n_out_channels": 1,
#                "augment": False,
#                "shuffle": False}
# validation_generator = DataGenerator(testList, **validation_data_params)

# Fit the model
if args.horovod:
    if hvd.rank() == 0:  # Only do validation and callbacks on chief
        model.fit_generator(training_generator,
                  epochs=args.epochs, verbose=1,
                  #validation_data=validation_generator,
                  validation_data=(imgs_test,msks_test),
                  callbacks=callbacks_list)
    else:
        model.fit_generator(training_generator,
                  epochs=args.epochs, verbose=0,
                  #validation_data=validation_generator,
                  validation_data=(imgs_test,msks_test),
                  callbacks=hvd_callbacks # Just do the horovod callbacks
                  )
else:
    model.fit_generator(training_generator,
              epochs=args.epochs, verbose=1,
              #validation_data=validation_generator,
              validation_data=(imgs_test,msks_test),
              callbacks=callbacks_list)

if args.horovod:
    if hvd.rank()==0:
        stop_time = time.time()
        print("\n\nTotal time = {:,.3f} seconds".format(stop_time - start_time))
        print("Stopped script on {}".format(datetime.datetime.now()))
else:
    stop_time = time.time()
    print("\n\nTotal time = {:,.3f} seconds".format(stop_time - start_time))
    print("Stopped script on {}".format(datetime.datetime.now()))
