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
import multiprocessing
import argparse
parser = argparse.ArgumentParser(
	description="Benchmark 3D and 2D Convolution Models",add_help=True)
parser.add_argument("--dim_length",
					type = int,
					default=16,
					help="Tensor cube length of side")
parser.add_argument("--num_channels",
					type = int,
					default=1,
					help="Number of channels")
parser.add_argument("--num_outputs",
					type = int,
					default=1,
					help="Number of outputs")

parser.add_argument("--bz",
					type = int,
					default=1,
					help="Batch size")

parser.add_argument("--lr",
					type = float,
					default=0.001,
					help="Learning rate")

parser.add_argument("--num_datapoints",
					type = int,
					default=1024,
					help="Number of datapoints")
parser.add_argument("--epochs",
					type = int,
					default=3,
					help="Number of epochs")
parser.add_argument("--intraop_threads",
					type = int,
					default=multiprocessing.cpu_count()-1, # All but one core
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
parser.add_argument("--D2",
					action="store_true",
					default=False,
					help="Use 2D model and images instead of 3D.")
parser.add_argument("--single_class_output",
					action="store_true",
					default=False,
					help="Use binary classifier instead of U-Net")
parser.add_argument("--mkl_verbose",
					action="store_true",
					default=False,
					help="Print MKL debug statements.")

args = parser.parse_args()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Get rid of the AVX, SSE warnings
if args.mkl_verbose:
	os.environ["MKL_VERBOSE"] = "1"  # Print out messages from MKL operations
os.environ["OMP_NUM_THREADS"] = str(args.intraop_threads)
os.environ["KMP_BLOCKTIME"] = str(args.blocktime)
os.environ["KMP_AFFINITY"] = "granularity=thread,compact,1,0"

import tensorflow as tf
from model import *
from tqdm import trange, tqdm
tqdm.monitor_interval = 0

print("\nArgs = {}".format(args))

if args.D2:  # Define shape of the tensors (2D)
	dims = (1,2)
	tensor_shape = [args.bz,
					args.dim_length,
					args.dim_length,
					args.num_channels]
	out_shape = [args.bz,
					args.dim_length,
					args.dim_length,
					args.num_outputs]
else:        # Define shape of the tensors (3D)
	dims=(1,2,3)
	tensor_shape = [args.bz,
					args.dim_length,
					args.dim_length,
					args.dim_length,
					args.num_channels]
	tensor_shape = [args.bz,
					args.dim_length,
					args.dim_length,
					args.dim_length,
					args.num_outputs]

# Optimize CPU threads for TensorFlow
config = tf.ConfigProto(
		inter_op_parallelism_threads=args.interop_threads,
		intra_op_parallelism_threads=args.intraop_threads)

sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)

global_step = tf.Variable(0, name="global_step", trainable=False)

# Define the shape of the input images
# For segmentation models, the label (mask) is the same shape.
img = tf.placeholder(tf.float32, shape=tensor_shape) # Input tensor

if args.single_class_output:
	truth = tf.placeholder(tf.float32, shape=(args.bz,args.num_outputs)) # Label tensor
else:
	truth = tf.placeholder(tf.float32, shape=tensor_shape) # Label tensor

# Define the model
# Predict the output mask

if args.single_class_output:
	if args.D2:    # 2D convnet model
		predictions = conv2D(img,
					   print_summary=args.print_model, n_out=args.num_outputs)
	else:			# 3D convet model
		predictions = conv3D(img,
					   print_summary=args.print_model, n_out=args.num_outputs)
else:

	if args.D2:    # 2D U-Net model
		predictions = unet2D(img,
					   use_upsampling=args.use_upsampling,
					   print_summary=args.print_model, n_out=args.num_outputs)
	else:			# 3D U-Net model
		predictions = unet3D(img,
					   use_upsampling=args.use_upsampling,
					   print_summary=args.print_model, n_out=args.num_outputs)

#  Performance metrics for model
if args.single_class_output:
	loss = tf.losses.sigmoid_cross_entropy(truth, predictions)
	metric_score = tf.metrics.mean_squared_error(truth, predictions)
else:
	loss = dice_coef_loss(truth, predictions, dims)  # Loss is the dice between mask and prediction
	metric_score = dice_coef(truth, predictions, dims)

train_op = tf.train.AdamOptimizer(args.lr).minimize(loss, global_step=global_step)

# Just feed completely random data in for the benchmark testing
imgs = np.random.rand(*tensor_shape)

if args.single_class_output:
	truths = np.random.rand(args.bz, args.num_outputs)
else:
	truths = np.random.rand(*tensor_shape)

# Initialize all variables
init_op = tf.global_variables_initializer()
init_l = tf.local_variables_initializer() # For TensorFlow metrics
sess.run(init_op)
sess.run(init_l)

# Set up trace for operations
run_metadata = tf.RunMetadata()
run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

for epoch in range(args.epochs):

	# Same number of sample to process regardless of batch size
	# So if we have a larger batch size we can take fewer steps.
	total_steps = args.num_datapoints//args.bz
	progressbar = trange(total_steps) # tqdm progress bar
	last_step = 0
	for i in range(total_steps):
		feed_dict = {img: imgs, truth:truths}

		history, loss_v, metric_v, this_step = \
				sess.run([train_op, loss, metric_score, global_step],
				feed_dict=feed_dict,
				options=run_options, run_metadata=run_metadata)

		# Print the loss and dice metric in the progress bar.
		if args.single_class_output:
			progressbar.set_description(
						"Epoch {}/{}: (loss={:.4f}, MSE={:.4f})".format(
						epoch+1, args.epochs, loss_v, metric_v[1]))
		else:
			progressbar.set_description(
						"Epoch {}/{}: (loss={:.4f}, dice={:.4f})".format(
						epoch+1, args.epochs, loss_v, metric_v))
		progressbar.update(this_step-last_step)
		last_step = this_step

'''
Save the training timeline
'''
from tensorflow.python.client import timeline

timeline_filename = "./timeline_trace.json"
fetched_timeline = timeline.Timeline(run_metadata.step_stats)
chrome_trace = fetched_timeline.generate_chrome_trace_format()
with open(timeline_filename, "w") as f:
	print("Saved Tensorflow trace to: {}".format(timeline_filename))
	print("To view the trace:\n(1) Open Chrome browser.\n"
	"(2) Go to this url -- chrome://tracing\n"
	"(3) Click the load button.\n"
	"(4) Load the file {}.".format(timeline_filename))
	f.write(chrome_trace)
