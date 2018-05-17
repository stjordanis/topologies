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
import os
import argparse
import psutil
import time
import datetime
import tensorflow as tf
from model import *

parser = argparse.ArgumentParser(
	description="Train 3D U-Net model",add_help=True)
parser.add_argument("--bz",
					type = int,
					default=1,
					help="Batch size")
parser.add_argument("--lr",
					type = float,
					default=0.001,
					help="Learning rate")
parser.add_argument("--epochs",
					type = int,
					default=20,
					help="Number of epochs")
parser.add_argument("--intraop_threads",
					type = int,
					default=psutil.cpu_count(logical=False),
					help="Number of intraop threads")
parser.add_argument("--interop_threads",
					type = int,
					default=2,
					help="Number of interop threads")
parser.add_argument("--blocktime",
					type = int,
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

args = parser.parse_args()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Get rid of the AVX, SSE warnings
os.environ["OMP_NUM_THREADS"] = str(args.intraop_threads)
os.environ["KMP_BLOCKTIME"] = str(args.blocktime)
os.environ["KMP_AFFINITY"] = "granularity=thread,compact,1,0"


print("Started script on {}".format(datetime.datetime.now()))

print("args = {}".format(args))
print("OS: {}".format(os.system("uname -a")))
print("TensorFlow version: {}".format(tf.__version__))

import keras as K

print("Keras API version: {}".format(K.__version__))

# Optimize CPU threads for TensorFlow
config = tf.ConfigProto(
		inter_op_parallelism_threads=args.interop_threads,
		intra_op_parallelism_threads=args.intraop_threads)

sess = tf.Session(config=config)
K.backend.set_session(sess)

"""
Load the data here. If it is too big, then you
may need fit_generator.
"""
imgs_train = np.random.rand(200,128,128,128,1)
msks_train = np.random.rand(200,128,128,128,1)

imgs_test = np.random.rand(45,128,128,128,1)
msks_test = np.random.rand(45,128,128,128,1)

input_shape = imgs_train.shape[1:]

model = unet_3d(input_shape,
				args.use_upsampling,
				args.lr,
				1,  # single channel (greyscale)
				0.2,
				True)

start_time = time.time()

# Save best model to hdf5 file
checkpoint = K.callbacks.ModelCheckpoint("./saved_model",
							 verbose=1,
							 save_best_only=True)

# Stop early if loss hasn't improved in 4 epochs
earlystopping = K.callbacks.EarlyStopping(monitor=[dice_coef_loss],
											  patience=4, verbose=1)

# TensorBoard
tb_logs = K.callbacks.TensorBoard(log_dir='./tensorboard_logs')

callbacks_list = [checkpoint, earlystopping, tb_logs]

# Fit the model
model.fit(x=imgs_train, y=msks_train,
		  batch_size=args.bz,
		  epochs=args.epochs, verbose=1,
		  validation_data=(imgs_test, msks_test),
		  callbacks=callbacks_list,
		  shuffle=True)

stop_time = time.time()

print("\n\nTotal time = {:,.3f} seconds".format(stop_time - start_time))

print("Stopped script on {}".format(datetime.datetime.now()))
