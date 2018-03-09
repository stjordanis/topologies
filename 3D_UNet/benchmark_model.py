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

import os
import argparse
parser = argparse.ArgumentParser(description="Benchmark 3D U-Net",add_help=True)
parser.add_argument("--dim_length",
					type = int,
					default=16,
					help="Tensor cube length of side")
parser.add_argument("--num_channels",
					type = int,
					default=1,
					help="Number of channels")

parser.add_argument("--bz",
					type = int,
					default=10,
					help="Batch size")

parser.add_argument("--lr",
					type = float,
					default=0.001,
					help="Learning rate")

parser.add_argument("--num_datapoints",
					type = int,
					default=31000,
					help="Number of datapoints")
parser.add_argument("--epochs",
					type = int,
					default=3,
					help="Number of epochs")
parser.add_argument("--intraop_threads",
					type = int,
					default=60,
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

import tensorflow as tf
from model import define_model, dice_coef_loss, dice_coef
from model import sensitivity, specificity
from tqdm import trange

import numpy as np

print("\nArgs = {}".format(args))

# Optimize CPU threads for TensorFlow
config = tf.ConfigProto(
		inter_op_parallelism_threads=args.interop_threads,
		intra_op_parallelism_threads=args.intraop_threads)

sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)

global_step = tf.Variable(0, name="global_step", trainable=False)

# Define the shape of the input images
# For segmentation models, the label (mask) is the same shape.
shape = (None, args.dim_length,
 		            args.dim_length,
 		            args.dim_length,
 		            args.num_channels)
img = tf.placeholder(tf.float32, shape=shape) # Input tensor
msk = tf.placeholder(tf.float32, shape=shape) # Label tensor

# Define the model
# Predict the output mask
preds = define_model(img, learning_rate=args.lr,
					 use_upsampling=args.use_upsampling,
					 print_summary=args.print_model)

#  Performance metrics for model
loss = dice_coef_loss(msk, preds)  # Loss is the dice between mask and prediction
dice_score = dice_coef(msk, preds)
sensitivity_score = sensitivity(msk, preds)
specificity_score = specificity(msk, preds)

train_op = tf.train.AdamOptimizer(args.lr).minimize(loss, global_step=global_step)

# Just feed completely random data in for the benchmark testing
imgs = np.random.rand(args.bz, args.dim_length,
			args.dim_length,
			args.dim_length,
			args.num_channels)
msks = imgs + np.random.rand(args.bz, args.dim_length,
			args.dim_length,
			args.dim_length,
			args.num_channels)

# Initialize all variables
init_op = tf.global_variables_initializer()
sess.run(init_op)

# Same number of sample to process regardless of batch size
# So if we have a larger batch size we can take fewer steps.
total_steps = args.num_datapoints//args.bz
progressbar = trange(total_steps) # tqdm progress bar
last_step = 0
for i in range(total_steps):
	feed_dict = {img: imgs, msk:msks}

	history, loss_v, dice_v, sensitivity_v, specificity_v, this_step = \
			sess.run([train_op, loss, dice_score,
			sensitivity_score, specificity_score, global_step],
			feed_dict=feed_dict)

	# Print the loss and dice metric in the progress bar.
	progressbar.set_description(
				"(loss={:.4f}, dice={:.4f})".format(loss_v, dice_v))
	progressbar.update(this_step-last_step)
	last_step = this_step
