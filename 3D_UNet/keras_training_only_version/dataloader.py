#!/usr/bin/python

# ----------------------------------------------------------------------------
# Copyright 2019 Intel
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

from imports import *  # All of the common imports

import os
import ntpath

import json

import nibabel as nib

class DataGenerator(K.utils.Sequence):
    """
    Generates data for Keras/TensorFlow

    This uses the Keras Sequence which is a better data pipeline.
    It will allow for multiple data generator processes and
    batch pre-fetching.

    If you have a different type of dataset, you'll just need to
    change the loading code in self.__data_generation to return
    the correct image and label.

    """

    def __init__(self,
                 isTraining,     # Boolean: Is this train or test set
                 data_path,    # File path for data
                 train_test_split=0.85, # Train test split
                 batch_size=8,  # batch size
                 dim=(128, 128, 128),  # Dimension of images/masks
                 n_in_channels=1,  # Number of channels in image
                 n_out_channels=1,  # Number of channels in mask
                 shuffle=True,  # Shuffle list after each epoch
                 augment=False,   # Augment images
                 seed=816):     # Seed for random number generator
        """
        Initialization
        """
        self.data_path = data_path
        self.isTraining = isTraining
        self.dim = dim
        self.batch_size = batch_size
        self.train_test_split = train_test_split

        self.n_in_channels = n_in_channels
        self.n_out_channels = n_out_channels
        self.shuffle = shuffle
        self.augment = augment

        self.seed = seed
        self.list_IDs = self.get_file_list()

        self.on_epoch_end()   # Generate the sequence

        self.num_batches = self.__len__()

        # Determine if axes are equal and can be rotated
        # If the axes aren't equal then we can't rotate them.
        equal_dim_axis = []
        for idx in range(0, len(dim)):
            for jdx in range(idx+1, len(dim)):
                if dim[idx] == dim[jdx]:
                    equal_dim_axis.append([idx, jdx]) # Valid rotation axes
        self.dim_to_rotate = equal_dim_axis

    def get_length(self):
        return len(self.list_IDs)

    def get_file_list(self):
        """
        Get list of the files from the BraTS raw data
        Split into training and testing sets.
        """
        json_filename = os.path.join(self.data_path, "dataset.json")

        try:
            with open(json_filename, "r") as fp:
                experiment_data = json.load(fp)
        except IOError as e:
            print("File {} doesn't exist. It should be part of the "
                  "Decathlon directory".format(json_filename))

        # Print information about the Decathlon experiment data
        print("*"*30)
        print("="*30)
        print("Dataset name:        ", experiment_data["name"])
        print("Dataset description: ", experiment_data["description"])
        print("Tensor image size:   ", experiment_data["tensorImageSize"])
        print("Dataset release:     ", experiment_data["release"])
        print("Dataset reference:   ", experiment_data["reference"])
        print("Input channels:      ", experiment_data["modality"])
        print("Output labels:       ", experiment_data["labels"])
        print("Dataset license:     ", experiment_data["licence"])  # sic
        print("="*30)
        print("*"*30)

        """
        Randomize the file list. Then separate into training and
        validation lists. We won't use the testing set since we
        don't have ground truth masks for this.
        """
        numFiles = experiment_data["numTraining"]
        idxList = np.arange(numFiles)  # List of file indices

        self.imgFiles = {}
        self.mskFiles = {}

        for idx in idxList:
            self.imgFiles[idx] = os.path.join(self.data_path,
                      experiment_data["training"][idx]["image"])
            self.mskFiles[idx] = os.path.join(self.data_path,
                      experiment_data["training"][idx]["label"])

        np.random.seed(self.seed)
        randomIdx = np.random.random(numFiles)  # List of random numbers
        # Random number go from 0 to 1. So anything above
        # self.train_split is in the validation list.
        trainIdx = idxList[randomIdx < self.train_test_split]
        testIdx = idxList[randomIdx >= self.train_test_split]

        if self.isTraining:
            print("Number of training MRIs = {}".format(len(trainIdx)))
            return trainIdx
        else:
            print("Number of test MRIs = {}".format(len(testIdx)))
            return testIdx

    def __len__(self):
        """
        The number of batches per epoch
        """
        return len(self.list_IDs) // self.batch_size

    def __getitem__(self, index):
        """
        Generate one batch of data
        """
        # Generate indicies of the batch
        indexes = np.sort(
            self.indexes[index*self.batch_size:(index+1)*self.batch_size])

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def get_batch(self, index):
        """
        Public method to get one batch of data
        """
        return self.__getitem__(index)

    def get_batch_fileIDs(self, index):
        """
        Get the original filenames for the batch at this index
        """
        indexes = np.sort(
            self.indexes[index*self.batch_size:(index+1)*self.batch_size])
        fileIDs = {}

        for idx, fileIdx in enumerate(indexes):
            name = self.imgFiles[fileIdx]
            filename = ntpath.basename(name) # Strip all but filename
            filename = os.path.splitext(filename)[0]
            fileIDs[idx] = os.path.splitext(filename)[0]

        return fileIDs

    def on_epoch_end(self):
        """
        Updates indices after each epoch
        If shuffle is true, then it will shuffle the training set
        after every epoch.
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def crop_img(self, img, msk, randomize=True):
        """
        Crop the image and mask
        """

        slices = []

        # Only randomize half when asked
        randomize = randomize and (np.random.rand() > 0.5)

        for idx in range(len(self.dim)):  # Go through each dimension

            cropLen = self.dim[idx]
            imgLen = img.shape[idx]

            start = (imgLen-cropLen)//2

            ratio_crop = 0.20  # Crop up this this % of pixels for offset
            # Number of pixels to offset crop in this dimension
            offset = int(np.floor(start*ratio_crop))

            if offset > 0:
                if randomize:
                    start += np.random.choice(range(-offset, offset))
                    if ((start + cropLen) > imgLen):  # Don't fall off the image
                        start = (imgLen-cropLen)//2
            else:
                start = 0

            slices.append(slice(start, start+cropLen))

        slices.append(slice(0,self.n_in_channels)) # No slicing along channels

        return img[tuple(slices)], msk[tuple(slices)]

    def augment_data(self, img, msk):
        """
        Data augmentation
        Flip image and mask. Rotate image and mask.
        """

        if np.random.rand() > 0.5:
            # Random 0,1 (axes to flip)
            ax = np.random.choice(np.arange(len(self.dim)-1))
            img = np.flip(img, ax)
            msk = np.flip(msk, ax)

        elif (len(self.dim_to_rotate) > 0) and (np.random.rand() > 0.5):
            rot = np.random.choice([1, 2, 3])  # 90, 180, or 270 degrees

            # This will choose the axes to rotate
            # Axes must be equal in size
            random_axis = self.dim_to_rotate[np.random.choice(len(self.dim_to_rotate))]
            img = np.rot90(img, rot, axes=random_axis) # Rotate axes 0 and 1
            msk = np.rot90(msk, rot, axes=random_axis) # Rotate axes 0 and 1

        # elif np.random.rand() > 0.5:
        #     rot = np.random.choice([1, 2, 3])  # 90, 180, or 270 degrees
        #     axis = np.random.choice([0, 1]) # Axis to rotate through
        #     img = np.rot90(img, rot, axes=(axis,2))
        #     msk = np.rot90(msk, rot, axes=(axis,2))

        return img, msk

    def z_normalize_img(self, img):
        """
        Normalize the image so that the mean value for each image
        is 0 and the standard deviation is 1.
        """
        for channel in range(img.shape[-1]):

            img_temp = img[...,channel]
            img_temp = (img_temp - np.mean(img_temp)) / np.std(img_temp)

            img[...,channel] = img_temp

        return img

    def __data_generation(self, list_IDs_temp):
        """
        Generates data containing batch_size samples

        This just reads the list of filename to load.
        Change this to suit your dataset.
        """

        # Make empty arrays for the images and mask batches
        imgs = np.zeros((self.batch_size, *self.dim, self.n_in_channels))
        msks = np.zeros((self.batch_size, *self.dim, self.n_out_channels))

        for idx, fileIdx in enumerate(list_IDs_temp):

            img_temp = np.array(nib.load(self.imgFiles[fileIdx]).dataobj)

            """
            "modality": {
                 "0": "FLAIR",
                 "1": "T1w",
                 "2": "t1gd",
                 "3": "T2w"
            """
            if self.n_in_channels == 1:
                img = img_temp[:,:,:,[0]]  # FLAIR channel
            else:
                img = img_temp

            # Get mask data
            msk = np.array(nib.load(self.mskFiles[fileIdx]).dataobj)

            """
            "labels": {
                 "0": "background",
                 "1": "edema",
                 "2": "non-enhancing tumor",
                 "3": "enhancing tumour"}
             """
            # Combine all masks but background
            msk[msk > 0] = 1.0
            msk = np.expand_dims(msk, -1)

            # Take a crop of the patch_dim size
            img, msk = self.crop_img(img, msk, self.augment)

            img = self.z_normalize_img(img)  # Normalize the image

            # Data augmentation
            if self.augment and (np.random.rand() > 0.5):
                img, msk = self.augment_data(img, msk)

            imgs[idx, ] = img
            msks[idx, ] = msk

        return imgs, msks
