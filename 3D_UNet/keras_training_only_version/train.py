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
                    default=0.01,
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
                    default=psutil.cpu_count(logical=False)-2,
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
parser.add_argument("--saved_model",
                    default="./saved_model_no_horovod/3d_unet_brats2018.hdf5",
                    help="Save model to this path")

args = parser.parse_args()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Get rid of the AVX, SSE warnings
os.environ["OMP_NUM_THREADS"] = str(args.intraop_threads)
os.environ["KMP_BLOCKTIME"] = str(args.blocktime)
os.environ["KMP_AFFINITY"] = "granularity=thread,compact,1,0"

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


input_shape = [args.patch_dim, args.patch_dim, args.patch_dim, args.number_input_channels]


print_summary = args.print_model
verbose = 1

model, opt = unet_3d(input_shape=input_shape,
                use_upsampling=args.use_upsampling,
                learning_rate=args.lr,
                n_cl_in=args.number_input_channels,
                n_cl_out=1,  # single channel (greyscale)
                dropout=0.2,
                print_summary=print_summary)

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
tb_logs = K.callbacks.TensorBoard(log_dir=os.path.join(
    saved_model_directory, "tensorboard_logs"), update_freq="batch")

# Keep reducing learning rate if we get to plateau
reduce_lr = K.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2,
                              patience=5, min_lr=0.0001)

callbacks = [checkpoint, tb_logs, reduce_lr]

# Separate file lists into train and test sets
trainList, testList = get_file_list()
with open("trainlist.txt", "w") as f:
    for item in trainList:
        f.write("{}\n".format(item))

with open("testlist.txt", "w") as f:
    for item in testList:
        f.write("{}\n".format(item))

print("Number of training MRIs = {}".format(len(trainList)))
print("Number of test MRIs = {}".format(len(testList)))

seed = 816
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
                          "shuffle": False,
                          "seed": 816}
validation_generator = DataGenerator(testList, **validation_data_params)

# Fit the model
model.fit_generator(training_generator,
                    epochs=args.epochs, verbose=verbose,
                    validation_data=validation_generator,
                    callbacks=callbacks)


stop_time = time.time()
print("\n\nTotal time = {:,.3f} seconds".format(
    stop_time - start_time))
print("Stopped script on {}".format(datetime.datetime.now()))
