import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Get rid of the AVX, SSE warnings
os.environ["OMP_NUM_THREADS"] = "60"

import tensorflow as tf
from model import define_model
from tqdm import tqdm

import numpy as np

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

parser.add_argument("--num_datapoints",
					type = int,
					default=1000,
					help="Number of datapoints")
parser.add_argument("--epochs",
					type = int,
					default=3,
					help="Number of epochs")
args = parser.parse_args()

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

model = define_model(imgs, print_summary=True)

print(np.shape(imgs))
print(np.shape(msks))
model.fit(imgs, msks, batch_size=args.bz, epochs=args.epochs, verbose=1)
