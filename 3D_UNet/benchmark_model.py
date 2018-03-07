import os
import argparse
parser = argparse.ArgumentParser(description="Benchmark U-Net",add_help=True)
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
args = parser.parse_args()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Get rid of the AVX, SSE warnings
os.environ["OMP_NUM_THREADS"] = str(args.intraop_threads)
os.environ["KMP_BLOCKTIME"] = str(args.blocktime)
os.environ["KMP_AFFINITY"] = "granularity=thread,compact,1,0"

import tensorflow as tf
from model import define_model
from tqdm import tqdm

import numpy as np

print("Creating random dataset")
imgs = np.random.rand(args.num_datapoints, args.dim_length,
            args.dim_length,
            args.dim_length,
            args.num_channels)
msks = imgs + np.random.rand(args.num_datapoints, args.dim_length,
            args.dim_length,
            args.dim_length,
            args.num_channels)
print("Finished creating random dataset")
print("Input images shape = {}".format(np.shape(imgs)))
print("Masks shape = {}".format(np.shape(msks)))

# Optimize CPU threads for TensorFlow
config = tf.ConfigProto(
		inter_op_parallelism_threads=args.interop_threads,
		intra_op_parallelism_threads=args.intraop_threads)

sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)

model = define_model(imgs, learning_rate=args.lr, print_summary=True)

tb_callback = tf.keras.callbacks.TensorBoard(log_dir='./tb_logs',
							histogram_freq=0,
							batch_size=32,
							write_graph=True,
							write_grads=False,
							write_images=True)

model.fit(imgs, msks, batch_size=args.bz, epochs=args.epochs, verbose=1,
			callbacks=[tb_callback])
