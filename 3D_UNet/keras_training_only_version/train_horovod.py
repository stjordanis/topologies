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

import keras as K
import numpy as np

import random
import os
import sys
import argparse
import psutil
import time
import datetime
import tensorflow as tf
from model import *

from dataloader import DataGenerator

import horovod.keras as hvd
hvd.init()

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
                    default=0.004,
                    help="Learning rate")
parser.add_argument("--train_test_split",
                    type=float,
                    default=0.85,
                    help="Train test split (0-1)")
parser.add_argument("--epochs",
                    type=int,
                    default=35,
                    help="Number of epochs")
parser.add_argument("--intraop_threads",
                    type=int,
                    default=psutil.cpu_count(logical=False)-4,
                    help="Number of intraop threads")
parser.add_argument("--interop_threads",
                    type=int,
                    default=1,
                    help="Number of interop threads")
parser.add_argument("--blocktime",
                    type=int,
                    default=1,
                    help="Block time for CPU threads")
parser.add_argument("--number_input_channels",
                    type=int,
                    default=1,
                    help="Number of input channels")
parser.add_argument("--print_model",
                    action="store_true",
                    default=False,
                    help="Print the summary of the model layers")
parser.add_argument("--use_upsampling",
                    action="store_true",
                    default=False,
                    help="Use upsampling instead of transposed convolution")
datapath = "../../../data/Brats2018/"
parser.add_argument("--data_path",
                    default=datapath,
                    help="Root directory for BraTS 2018 dataset")

if hvd.rank() == 0:
    model_filename = "./saved_model_{}workers/3d_unet_brats2018.hdf5".format(hvd.size())
else:
    model_filename = "./saved_model_{}workers/3d_unet_brats2018_worker{}.hdf5".format(hvd.size(),hvd.rank())
parser.add_argument("--saved_model",
                    default=model_filename,
                    help="Save model to this path")

args = parser.parse_args()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Get rid of the AVX, SSE warnings
os.environ["OMP_NUM_THREADS"] = str(args.intraop_threads)
os.environ["KMP_BLOCKTIME"] = str(args.blocktime)
os.environ["KMP_AFFINITY"] = "granularity=thread,compact,1,0"

if hvd.rank() == 0:
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


input_shape = [args.patch_dim, args.patch_dim, args.patch_dim,
               args.number_input_channels]


if (hvd.rank() == 0):
    print_summary = args.print_model
    verbose = 1
else:
    print_summary = args.print_model
    verbose = 0


model, opt = unet_3d(input_shape=input_shape,
                use_upsampling=args.use_upsampling,
                n_cl_in=args.number_input_channels,
                learning_rate=args.lr*hvd.size(),
                n_cl_out=1,  # single channel (greyscale)
                dropout=0.2,
                print_summary=print_summary)

opt = hvd.DistributedOptimizer(opt)

model.compile(optimizer=opt,
              #loss=[combined_dice_ce_loss],
              loss=[dice_coef_loss],
              metrics=[dice_coef, "accuracy",
                       sensitivity, specificity])

start_time = time.time()

# Save best model to hdf5 file
saved_model_directory = os.path.dirname(args.saved_model)
try:
    os.stat(saved_model_directory)
except:
    os.mkdir(saved_model_directory)

# if os.path.isfile(args.saved_model):
#     model.load_weights(args.saved_model)

checkpoint = K.callbacks.ModelCheckpoint(args.saved_model,
                                         verbose=verbose,
                                         save_best_only=True)

# TensorBoard
if (hvd.rank() == 0):
    tb_logs = K.callbacks.TensorBoard(log_dir=os.path.join(
        saved_model_directory, "tensorboard_logs"), update_freq="batch")
else:
    tb_logs = K.callbacks.TensorBoard(log_dir=os.path.join(
        saved_model_directory, "tensorboard_logs_worker{}".format(hvd.rank())), update_freq="batch")

# NOTE:
# Horovod talks about having callbacks for rank 0 and callbacks
# for other ranks. For example, they recommend only doing checkpoints
# and tensorboard on rank 0. However, if there is a signficant time
# to execute tensorboard update or checkpoint update, then
# this might cause an issue with rank 0 not returning in time.
# My thought is that all ranks need to have essentially the same
# time taken for each rank.
callbacks = [
    # Horovod: broadcast initial variable states from
    # rank 0 to all other processes.
    # This is necessary to ensure consistent initialization
    # of all workers when
    # training is started with random weights or
    # restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),

    # Horovod: average metrics among workers at the end of every epoch.
    #
    # Note: This callback must be in the list before the ReduceLROnPlateau,
    # TensorBoard or other metrics-based callbacks.
    hvd.callbacks.MetricAverageCallback(),

    # Horovod: using `lr = 1.0 * hvd.size()` from the very
    # beginning leads to worse final
    # accuracy. Scale the learning rate
    # `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
    # the first five epochs. See https://arxiv.org/abs/1706.02677
    # for details.
    hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=3, verbose=verbose),

    # Reduce the learning rate if training plateaus.
    K.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.6,
                                  verbose=verbose,
                                  patience=5, min_lr=0.0001),
    tb_logs,  # we need this here otherwise tensorboard delays rank 0
    checkpoint
]

# Separate file lists into train and test sets
trainList, testList = get_file_list()
with open("trainlist.txt", "w") as f:
    for item in trainList:
        f.write("{}\n".format(item))

with open("testlist.txt", "w") as f:
    for item in testList:
        f.write("{}\n".format(item))


if hvd.rank() == 0:
    print("Number of training MRIs = {}".format(len(trainList)))
    print("Number of test MRIs = {}".format(len(testList)))

# Run the script  "load_brats_images.py" to generate these Numpy data files
#imgs_test = np.load(os.path.join(sys.path[0],"imgs_test_3d.npy"))
#msks_test = np.load(os.path.join(sys.path[0],"msks_test_3d.npy"))

seed = hvd.rank()  # Make sure each worker gets different random seed

training_data_params = {"dim": (args.patch_dim, args.patch_dim, args.patch_dim),
                        "batch_size": args.bz,
                        "n_in_channels": args.number_input_channels,
                        "n_out_channels": 1,
                        "augment": True,
                        "shuffle": True,
                        "seed": seed}

training_generator = DataGenerator(trainList, **training_data_params)

validation_data_params = {"dim": (args.patch_dim, args.patch_dim, args.patch_dim),
                          "batch_size": args.bz,
                          "n_in_channels": args.number_input_channels,
                          "n_out_channels": 1,
                          "augment": False,
                          "shuffle": True,
                          "seed": 816}
validation_generator = DataGenerator(testList, **validation_data_params)

# Fit the model
steps_per_epoch = max(3, len(trainList)//(args.bz*hvd.size()))
validation_steps = max(3,3*len(trainList)//(args.bz*hvd.size()))
model.fit_generator(training_generator,
                    steps_per_epoch=steps_per_epoch,
                    epochs=args.epochs, verbose=verbose,
                    validation_data=validation_generator,
		            #validation_steps=validation_steps,
                    callbacks=callbacks)

if hvd.rank() == 0:
    stop_time = time.time()
    print("\n\nTotal time = {:,.3f} seconds".format(
        stop_time - start_time))
    print("Stopped script on {}".format(datetime.datetime.now()))
