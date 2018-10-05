import numpy as np
import random
import os
import argparse
import psutil
import time
import datetime
import tensorflow as tf
from model import *
import nibabel as nib

parser = argparse.ArgumentParser(
    description="Train 3D U-Net model", add_help=True)
parser.add_argument("--bz",
                    type=int,
                    default=1,
                    help="Batch size")
parser.add_argument("--patch_dim",
                    type=int,
                    default=128,
                    help="Size of the 3D patch")
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
parser.add_argument("--model",
                    default="3d_unet_brat2018_dice82.hdf5",
                    help="Trained model to load")

args = parser.parse_args()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Get rid of the AVX, SSE warnings
os.environ["OMP_NUM_THREADS"] = str(args.intraop_threads)
os.environ["KMP_BLOCKTIME"] = str(args.blocktime)
os.environ["KMP_AFFINITY"] = "granularity=thread,compact,1,0"

# Optimize CPU threads for TensorFlow
config = tf.ConfigProto(
    inter_op_parallelism_threads=args.interop_threads,
    intra_op_parallelism_threads=args.intraop_threads)

sess = tf.Session(config=config)
K.backend.set_session(sess)

model = K.models.load_model(os.path.join("saved_model",args.model),
                            custom_objects={"dice_coef":dice_coef,
                            "dice_coef_loss":dice_coef_loss,
                            "sensitivity":sensitivity,
                            "specificity":specificity,
                            "combined_dice_ce_loss":combined_dice_ce_loss})

print("Loading images and masks from test set")
print("imgs_test_3d.npy, msks_test_3d.npy")
imgs = np.load("imgs_test_3d.npy")
msks = np.load("msks_test_3d.npy")

m = model.evaluate(imgs, msks, batch_size=args.bz, verbose=1)

print("Test metrics")
print("============")
i = 0
for name in model.metrics_names:
    print("{} = {:.4f}".format(name, m[i]))
    i += 1

save_directory = "predictions_directory"
try:
    os.stat(save_directory)
except:
    os.mkdir(save_directory)

print("Predicting masks")
preds = model.predict(imgs, args.bz)
np.save(os.path.join(save_directory, "msks_pred_3d.npy"), preds)

"""

Save the predictions as Nifti files so that we can
display them on a 3D viewer.

"""
import nibabel as nib
from tqdm import tqdm

print("Saving Nifti predictions to {}".format(save_directory))
for idx in tqdm(range(preds.shape[0])):

    img = nib.Nifti1Image(imgs[idx,:,:,:,0], np.eye(4))
    img.to_filename(os.path.join(save_directory,"img{}.nii.gz".format(idx)))

    msk = nib.Nifti1Image(msks[idx,:,:,:,0], np.eye(4))
    msk.to_filename(os.path.join(save_directory,"msk{}.nii.gz".format(idx)))

    pred = nib.Nifti1Image(preds[idx,:,:,:,0], np.eye(4))
    pred.to_filename(os.path.join(save_directory,"pred{}.nii.gz".format(idx)))
