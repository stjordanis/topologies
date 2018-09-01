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

import os
import nibabel as nib
import numpy.ma as ma
import numpy as np
from tqdm import tqdm
import h5py

# This is essential to make sure we get the same sequence each time
np.random.seed(816) # seed with same number

import argparse

parser = argparse.ArgumentParser(
               description="Convert BraTS raw Nifti data "
               "(https://www.med.upenn.edu/sbia/brats2018/data.html) "
               "files to Numpy data files",
               add_help=True)

parser.add_argument("--data_path",
                    default="/mnt/data/medical/brats/Brats2018/MICCAI_BraTS_2018_Data_Training/",
                    help="Path to the raw BraTS datafiles")
parser.add_argument("--save_path",
                    default="/mnt/data/medical/brats/Brats2018/",
                    help="Folder to save Numpy data files")
parser.add_argument("--resize", type=int, default=128,
                    help="Resize height and width to this size. "
                    "Original size = 240")
parser.add_argument("--rotate", type=int, default=3,
                    help="Number of counter-clockwise, 90 degree rotations")
parser.add_argument("--split", type=float, default=0.85,
                    help="Train/test split ratio")
parser.add_argument("--save_interval", type=int, default=25,
                    help="Interval between images to save file.")

args = parser.parse_args()

print("Converting BraTS raw Nifti data files to training and testing" \
        " Numpy data files.")
print(args)

def parse_segments(seg):

    # Each channel corresponds to a different region of the tumor,
    #  decouple and stack these

    msks_parsed = []
    for slice in range(seg.shape[-1]):
        curr = seg[:, :, slice]
        GD = ma.masked_not_equal(curr, 4).filled(fill_value=0)
        edema = ma.masked_not_equal(curr, 2).filled(fill_value=0)
        necrotic = ma.masked_not_equal(curr, 1).filled(fill_value=0)
        none = ma.masked_not_equal(curr, 0).filled(fill_value=0)

        msks_parsed.append(np.dstack((none, necrotic, edema, GD)))

    # Replace all tumorous areas with 1 (previously marked as 1, 2 or 4)
    mask = np.asarray(msks_parsed)
    mask[mask > 0] = 1

    return mask


def parse_images(img):
    """
    Read the 3D images and stack the slices
    """

    slices = []
    for slice in range(img.shape[-1]):
        curr = img[:, :, slice]
        slices.append(curr)

    return np.asarray(slices)


def stack_img_slices(mode_track, stack_order):
    """
    Put final image channels in the order listed in stack_order
    """

    full_brain = []
    for slice in range(len(mode_track["t1"])):
        current_slice = []
        for mode in stack_order:
            current_slice.append(mode_track[mode][slice, :, :])
        full_brain.append(np.dstack(current_slice))

    # Normalize stacked images (inference will not work if
    #  this is not performed)
    stack = np.asarray(full_brain)
    stack = (stack - np.mean(stack))/np.std(stack)

    #stack = stack / np.max(stack)

    return stack


def resize_data(dataset, new_size):
    """
    Test/Train images must be the same size
    """

    start_index = (dataset.shape[1] - new_size)//2
    end_index = dataset.shape[1] - start_index

    if args.rotate != 0:
        resized = np.rot90(
            dataset[:, start_index:end_index, start_index:end_index:],
                    args.rotate, axes=(1, 2))
    else:
        resized = dataset[:, start_index:end_index, start_index:end_index:]

    return resized


##################################################################
##################################################################
# Preprocess the total files sizes
sizecounter = 0
for subdir, dir, files in os.walk(args.data_path):
    sizecounter += 1

scan_count = 0

save_dir = os.path.join(args.save_path, "{}x{}/".format(args.resize, args.resize))

# Create directory
try:
    os.makedirs(save_dir)
except OSError:
    if not os.path.isdir(save_dir):
        raise

outputfilename = os.path.join(save_dir, "processed_data.hdf5")

assert(not os.path.isfile(outputfilename)), \
    "Output file {} exists. " \
    " Please delete and re-run script.".format(outputfilename)

hdfFile = h5py.File(outputfilename, "w-")

for subdir, dir, files in tqdm(os.walk(args.data_path), total=sizecounter):

    # Ensure all necessary files are present
    file_root = subdir.split("/")[-1] + "_"
    extension = ".nii.gz"
    img_modes = ["t1", "t2", "flair", "t1ce"]
    need_file = [file_root + mode + extension for mode in img_modes]
    all_there = [(reqd in files) for reqd in need_file]
    if all(all_there):

        mode_track = {mode: [] for mode in img_modes}

        for file in files:

            if file.endswith("seg.nii.gz"):
                path = os.path.join(subdir, file)
                msk = np.array(nib.load(path).dataobj)
                msks_all = resize_data(parse_segments(msk), args.resize)

            if file.endswith("t1.nii.gz"):
                path = os.path.join(subdir, file)
                img = np.array(nib.load(path).dataobj)
                mode_track["t1"] = resize_data(parse_images(img), args.resize)

            if file.endswith("t2.nii.gz"):
                path = os.path.join(subdir, file)
                img = np.array(nib.load(path).dataobj)
                mode_track["t2"] = resize_data(parse_images(img), args.resize)

            if file.endswith("t1ce.nii.gz"):
                path = os.path.join(subdir, file)
                img = np.array(nib.load(path).dataobj)
                mode_track["t1ce"] = resize_data(parse_images(img), args.resize)

            if file.endswith("flair.nii.gz"):
                path = os.path.join(subdir, file)
                img = np.array(nib.load(path).dataobj)
                mode_track["flair"] = resize_data(parse_images(img), args.resize)

        imgs_all = np.asarray(stack_img_slices(mode_track, img_modes))

        if (scan_count == 0):
            """
            Train dataset
            """
            shapeImage = imgs_all.shape
            shapeMask = msks_all.shape
            maxshapeImage = (None, shapeImage[1],shapeImage[2],shapeImage[3])
            maxshapeMask = (None, shapeImage[1], shapeImage[2], shapeImage[3])

            imgHDF_train = hdfFile.create_dataset("images/train",
                                      data=imgs_all,
                                      dtype=float,
                                      maxshape=maxshapeImage)
            mskHDF_train = hdfFile.create_dataset("masks/train",
                                      data=msks_all,
                                      dtype=float,
                                      maxshape=maxshapeMask)
        elif (scan_count == 1):
            """
            Test dataset
            """
            shapeImage = imgs_all.shape
            shapeMask = msks_all.shape
            maxshapeImage = (None, shapeImage[1],shapeImage[2],shapeImage[3])
            maxshapeMask = (None, shapeImage[1], shapeImage[2], shapeImage[3])
            imgHDF_test = hdfFile.create_dataset("images/test",
                                      data=imgs_all,
                                      dtype=float,
                                      maxshape=maxshapeImage)
            mskHDF_test = hdfFile.create_dataset("masks/test",
                                      data=msks_all,
                                      dtype=float,
                                      maxshape=maxshapeMask)
        else:

            # Randomly split into train and test datasets.
            # At the beginning of the script we set the seed
            # to 816 so that it will always go through the same way and
            # produce the same one.
            if np.random.rand() < args.split:
                row = imgHDF_train.shape[0]
                extent = imgs_all.shape[0]
                imgHDF_train.resize(row+extent, axis=0) # Add new image
                imgHDF_train[row:(row+extent),:,:,:] = imgs_all

                mskHDF_train.resize(row+extent, axis=0) # Add new image
                mskHDF_train[row:(row+extent),:,:,:] = msks_all

            else:
                row = imgHDF_test.shape[0]
                extent = imgs_all.shape[0]
                imgHDF_test.resize(row+extent, axis=0) # Add new image
                imgHDF_test[row:(row+extent),:,:,:] = imgs_all

                mskHDF_test.resize(row+extent, axis=0) # Add new image
                mskHDF_test[row:(row+extent),:,:,:] = msks_all


        scan_count += 1


imgHDF_train.attrs["lshape"] = np.shape(imgHDF_train)
mskHDF_train.attrs["lshape"] = np.shape(mskHDF_train)

imgHDF_test.attrs["lshape"] = np.shape(imgHDF_test)
mskHDF_test.attrs["lshape"] = np.shape(mskHDF_test)

print("Processed scans saved to: {}".format(os.path.join(args.save_path,
            "brats2018_data.hdf5")))
print("Total scans processed: {}\nDone.".format(scan_count))
