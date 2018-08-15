#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: EPL-2.0
#

"""

Runs U-Net training on BraTS dataset

"""
import tensorflow as tf
from tensorflow import layers

from model import define_model
from data import load_datasets, get_batch

import os
from datetime import datetime

import settings


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("num_inter_threads", 2,
							"# inter op threads")
tf.app.flags.DEFINE_integer("num_threads", os.cpu_count(),
							"# intra op threads")

tf.app.flags.DEFINE_integer("total_steps", 1000,
							"Number of training steps")

tf.app.flags.DEFINE_integer("log_steps", 30,
							"Number of steps between logs")
tf.app.flags.DEFINE_integer("batch_size", 128,
							"Batch Size for Training")
tf.app.flags.DEFINE_string("logdir", "checkpoints_mnist",
							"Log directory")
tf.app.flags.DEFINE_boolean("no_horovod", False,
							"Don't use Horovod. Single node training only.")
tf.app.flags.DEFINE_float("learningrate", 0.0005,
							"Learning rate")
tf.app.flags.DEFINE_boolean("use_upsampling", settings.USE_UPSAMPLING,
						"True = Use upsampling; False = Use transposed convolution")

os.environ["KMP_BLOCKTIME"] = "0"
os.environ["KMP_AFFINITY"] = "granularity=thread,compact,1,0"
os.environ["OMP_NUM_THREADS"] = str(FLAGS.num_threads)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Get rid of the AVX, SSE warnings

config = tf.ConfigProto(intra_op_parallelism_threads=FLAGS.num_threads,
						inter_op_parallelism_threads=FLAGS.num_inter_threads)

tf.logging.set_verbosity(tf.logging.INFO)

if not FLAGS.no_horovod:
	import horovod.tensorflow as hvd

def main(_):

	start_time = datetime.now()
	tf.logging.info("Starting at: {}".format(start_time))
	tf.logging.info("Batch size: {} images per step".format(FLAGS.batch_size))

	# Load datasets
	imgs_train, msks_train, imgs_test, msks_test = load_datasets(FLAGS)


	if not FLAGS.no_horovod:
		# Initialize Horovod.
		hvd.init()

	# Define model
	model = define_model(imgs_train.shape, msks_train.shape, FLAGS)

	if not FLAGS.no_horovod:
		# Horovod: adjust learning rate based on number workers
		opt = tf.train.AdamOptimizer(FLAGS.learningrate * hvd.size())
	else:
		opt = tf.train.AdamOptimizer(FLAGS.learningrate)

	# Wrap optimizer with Horovod Distributed Optimizer.
	if FLAGS.no_horovod is None:
		opt = hvd.DistributedOptimizer(opt)

	global_step = tf.train.get_or_create_global_step()
	train_op = opt.minimize(model["loss"], global_step=global_step)

	if not FLAGS.no_horovod:
		last_step = FLAGS.total_steps // hvd.size()
	else:
		last_step = FLAGS.total_steps

	def formatter_log(tensors):
		if FLAGS.no_horovod:
			logstring= "Step {} of {}: " \
			   " training loss = {:.4f}," \
		       " training Dice = {:.4f}".format(tensors["step"],
			   last_step,
			   tensors["loss"], tensors["dice"])
		else:
			   logstring= "HOROVOD (Worker #{}), Step {} of {}: " \
			   " training loss = {:.4f}," \
   		       " training dice = {:.4f}".format(
			   hvd.rank(),
			   tensors["step"],
			   last_step,
   			   tensors["loss"], tensors["dice"])

		return logstring

	hooks = [

		tf.train.StopAtStepHook(last_step=last_step),

		# Prints the loss and step every log_steps steps
		tf.train.LoggingTensorHook(tensors={"step": global_step,
									"loss": model["loss"],
									"dice": model["metric_dice"]},
								   every_n_iter=FLAGS.log_steps,
								   formatter=formatter_log),
	]

	# Horovod: BroadcastGlobalVariablesHook broadcasts
	# initial variable states from rank 0 to all other
	# processes. This is necessary to ensure consistent
	# initialization of all workers when training is
	# started with random weights
	# or restored from a checkpoint.
	if not FLAGS.no_horovod:
		hooks.append(hvd.BroadcastGlobalVariablesHook(0))

		# Horovod: save checkpoints only on worker 0 to prevent other workers from
		# corrupting them.
		if hvd.rank() == 0:
			checkpoint_dir = "{}/{}-workers/{}".format(FLAGS.logdir,
							hvd.size(),
							datetime.now().strftime("%Y%m%d-%H%M%S"))
		else:
			checkpoint_dir = None

	else:
		checkpoint_dir = "{}/no_hvd/{}".format(FLAGS.logdir,
						datetime.now().strftime("%Y%m%d-%H%M%S"))

	# The MonitoredTrainingSession takes care of session initialization,
	# restoring from a checkpoint, saving to a checkpoint,
	# and closing when done or an error occurs.
	with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
										   hooks=hooks,
										   save_summaries_steps=FLAGS.log_steps,
										   log_step_count_steps=FLAGS.log_steps,
										   config=config) as mon_sess:
		while not mon_sess.should_stop():

			# Run a training step synchronously.
			image_, mask_ = get_batch(imgs_train,
									  msks_train,
									  FLAGS.batch_size)

			mon_sess.run(train_op, feed_dict={model["input"]: image_,
										  	  model["label"]: mask_})


	stop_time = datetime.now()
	tf.logging.info("Stopping at: {}".format(stop_time))
	tf.logging.info("Elapsed time was: {}".format(stop_time-start_time))

if __name__ == "__main__":

	tf.app.run()
