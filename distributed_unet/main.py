# These are the only things you need to change.
# Just replace the IP addresses with whatever machines you want to distribute over
# Then run this script on each of those machines.
"""
Usage:  python test_dist.py --ip=10.100.68.816 --is_sync=0
		for asynchronous TF
		python test_dist.py --ip=10.100.68.816 --is_sync=1
		for synchronous updates
		The IP address must match one of the ones in the list below. If not passed,
		then we"ll default to the current machine"s IP (which is usually correct unless you use OPA)
"""
import settings_dist as settings

ps_hosts = settings.PS_HOSTS
ps_ports = settings.PS_PORTS
worker_hosts = settings.WORKER_HOSTS
worker_ports = settings.WORKER_PORTS

ps_list = ["{}:{}".format(x, y) for x, y in zip(ps_hosts, ps_ports)]
worker_list = [
	"{}:{}".format(x, y) for x, y in zip(worker_hosts, worker_ports)
]
print("Distributed TensorFlow training")
print("Parameter server nodes are: {}".format(ps_list))
print("Worker nodes are {}".format(worker_list))

####################################################################

import numpy as np
import tensorflow as tf
import os
import socket

# Fancy progress bar
from tqdm import trange

from model import define_model, dice_coef_loss, dice_coef, sensitivity, specificity
from data import load_all_data, get_epoch
import multiprocessing
import subprocess
import signal

num_inter_op_threads = settings.NUM_INTER_THREADS
num_intra_op_threads = settings.NUM_INTRA_THREADS  #multiprocessing.cpu_count() // 2 # Use half the CPU cores

# Unset proxy env variable to avoid gRPC errors
del os.environ["http_proxy"]
del os.environ["https_proxy"]

# You can turn on the gRPC messages by setting the environment variables below
#os.environ["GRPC_VERBOSITY"]="DEBUG"
#os.environ["GRPC_TRACE"] = "all"

os.environ["KMP_BLOCKTIME"] = str(settings.BLOCKTIME)
os.environ["KMP_AFFINITY"] = "granularity=thread,compact,1,0"
os.environ["OMP_NUM_THREADS"] = str(num_intra_op_threads)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Get rid of the AVX, SSE warnings

# Define parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean("const_learningrate", settings.CONST_LEARNINGRATE,
							"Keep learning rate constant or exponentially decay")
tf.app.flags.DEFINE_float("learning_rate", settings.LEARNINGRATE,
						  "Initial learning rate.")

tf.app.flags.DEFINE_float("lr_fraction", settings.LR_FRACTION,
							"Learning rate fraction for decay")
tf.app.flags.DEFINE_integer("decay_steps", settings.DECAY_STEPS,
							"Number of steps for decay")


tf.app.flags.DEFINE_integer("is_sync", 1, "Synchronous updates?")
tf.app.flags.DEFINE_string("ip", socket.gethostbyname(socket.gethostname()),
						   "IP address of this machine")
tf.app.flags.DEFINE_integer("batch_size", settings.BATCH_SIZE,
							"Batch size of input data")
tf.app.flags.DEFINE_integer("epochs", settings.EPOCHS,
							"Number of epochs to train")

tf.app.flags.DEFINE_boolean("use_upsampling", settings.USE_UPSAMPLING,
							"True = Use upsampling; False = Use transposed convolution")

if (FLAGS.ip in ps_hosts):
	job_name = "ps"
	task_index = ps_hosts.index(FLAGS.ip)
elif (FLAGS.ip in worker_hosts):
	job_name = "worker"
	task_index = worker_hosts.index(FLAGS.ip)
else:
	print(
		"Error: IP {} not found in the worker or ps node list.\nUse --ip= to specify which machine this is.".
		format(FLAGS.ip))
	exit()


CHECKPOINT_DIRECTORY = settings.CHECKPOINT_DIRECTORY
if FLAGS.use_upsampling:
	method_up = "upsample2D"
else:
	method_up = "conv2DTranspose"
CHECKPOINT_DIRECTORY = CHECKPOINT_DIRECTORY + "/unet," + \
			"lr={},{},intra={},inter={}".format(FLAGS.learning_rate,
			method_up, num_intra_op_threads,
			num_inter_op_threads)


def main(_):

	config = tf.ConfigProto(
		inter_op_parallelism_threads=num_inter_op_threads,
		intra_op_parallelism_threads=num_intra_op_threads)

	run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
	run_metadata = tf.RunMetadata()  # For Tensorflow trace

	cluster = tf.train.ClusterSpec({"ps": ps_list, "worker": worker_list})

	server = tf.train.Server(cluster,
		  job_name=job_name,
		  task_index=task_index,
		  config=config)

	is_chief = (task_index == 0)  # Am I the chief node (always task 0)

	if job_name == "ps":

		print("*" * 30)
		print("\nParameter server #{} on {}.\n\n" \
		 "Waiting on workers to finish.\n\nPress CTRL-\\ to terminate early.\n"  \
		 .format(task_index, ps_hosts[task_index]))
		print("*" * 30)

		server.join()

	elif job_name == "worker":

		if is_chief:
			print("I am the chief worker {} with task #{}".format(
				worker_hosts[task_index], task_index))
		else:
			print("I am worker {} with task #{}".format(
				worker_hosts[task_index], task_index))

		if len(ps_list) > 0:

			setDevice = tf.train.replica_device_setter(
					 worker_device="/job:worker/task:{}".format(task_index),
					 ps_tasks=len(ps_hosts),
					 cluster=cluster)
		else:
			setDevice = "/cpu:0"  # No parameter server so put variables on chief worker

		with tf.device(setDevice):

			# Load the data
			imgs_train, msks_train, imgs_test, msks_test = load_all_data()
			train_length = imgs_train.shape[0]  # Number of train datasets
			test_length  = imgs_test.shape[0]   # Number of test datasets

			"""
			BEGIN: Define our model
			All of the model definitions are in the file model.py
			"""
			model = define_model(FLAGS, imgs_train.shape[1:], imgs_train.shape[1:], len(worker_hosts))
			"""
			END: Define our model
			"""

			print("Loading epoch")
			epoch = get_epoch(FLAGS.batch_size, imgs_train, msks_train)
			num_batches = len(epoch)
			print("Loaded")


		# Session
		# The StopAtStepHook handles stopping after running given steps.
		# We'll just set the number of steps to be the # of batches * epochs
		stop_hook = tf.train.StopAtStepHook(last_step=num_batches * FLAGS.epochs)

		# Only the chief does the summary
		if is_chief:
			summary_op = tf.summary.merge_all()
			summary_hook = tf.train.SummarySaverHook(save_secs=120,
							output_dir=CHECKPOINT_DIRECTORY, summary_op=summary_op)
			hooks = [stop_hook, summary_hook]
		else:
			summary_op = None
			hooks = [stop_hook]

		# For synchronous SGD training.
		# This creates the hook for the MonitoredTrainingSession
		# So that the worker nodes will wait for the next training step
		# (which is signaled by the chief worker node)
		if FLAGS.is_sync:
			sync_replicas_hook = model["optimizer"].make_session_run_hook(is_chief)
			hooks.append(sync_replicas_hook)

		with tf.train.MonitoredTrainingSession(master = server.target,
				is_chief=is_chief,
				config=config,
				hooks=hooks,
				checkpoint_dir=CHECKPOINT_DIRECTORY,
				stop_grace_period_secs=10) as sess:

			progressbar = trange(num_batches * FLAGS.epochs)
			step = 0

			while not sess.should_stop():

				batch_idx = step % num_batches # Which batch is the epoch?

				data = epoch[batch_idx, 0]
				labels = epoch[batch_idx, 1]

				# For n workers, break up the batch into n sections
				# Send each worker a different section of the batch
				data_range = int(FLAGS.batch_size / len(worker_hosts))
				start = data_range * task_index
				end = start + data_range

				feed_dict = {model["input"]: data[start:end], model["label"]: labels[start:end]}

				history, loss_v, dice_v, step = sess.run(
					[model["train_op"], model["loss"], model["metric_dice"], model["global_step"]],
					feed_dict=feed_dict)

				# Print the loss and dice metric in the progress bar.
				progressbar.set_description(
					"(loss={:.3f}, dice={:.3f})".format(loss_v, dice_v))
				progressbar.n = step

		print("\n\nFinished work on this node.")


if __name__ == "__main__":

	tf.app.run()
