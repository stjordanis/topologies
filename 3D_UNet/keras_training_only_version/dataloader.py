
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

import keras as K
import numpy as np
import os

import nibabel as nib


class DataGenerator(K.utils.Sequence):
    """
    Generates data for Keras/TensorFlow

    Code based on https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

    This uses the Keras Sequence which is better for multiprocessing.
    The main input the dataloader is a list of filenames containing
    the images (MRIs) to load. In the case of BraTS, the images and masks
    have the same name but a different suffix. For example, the FLAIR image
    could be "MRI1234_flair.nii.gz" and the corresponding mask would be
    "MRI1234_seg.nii.gz".

    If you have a different type of dataset, you'll just need to
    change the loading code in self.__data_generation to return
    the correct image and label.

    """

    def __init__(self,
                 list_IDs,     # List of file names for raw images/masks
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
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_in_channels = n_in_channels
        self.n_out_channels = n_out_channels
        self.shuffle = shuffle
        self.augment = augment

        np.random.seed(seed)
        self.on_epoch_end()   # Generate the sequence

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

            if randomize:
                start += np.random.choice(range(-offset, offset))
                if ((start + cropLen) > imgLen):  # Don't fall off the image
                    start = (imgLen-cropLen)//2

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

        elif np.random.rand() > 0.5:
            rot = np.random.choice([1, 2, 3])  # 90, 180, or 270 degrees
            axis = np.random.choice([0, 1]) # Axis to rotate through
            img = np.rot90(img, rot, axes=(axis,2))
            msk = np.rot90(msk, rot, axes=(axis,2))

        return img, msk

    def z_normalize_img(self, img):
        """
        Normalize the image so that the mean value for each image
        is 0 and the standard deviation is 1.
        """
        for channel in range(img.shape[-1]):

            img_temp = img[...,channel]
            img_temp = (img_temp - np.mean(img_temp)) / np.std(img_temp)

            # Clip between -5 and 5
            # Based on  Isensee et al., 2017
            # https://arxiv.org/pdf/1802.10508v1.pdf
            img_temp[img_temp > 5] = 5
            img_temp[img_temp < -5] = -5

            # Translate positive and normalize between 0 and 1
            img_temp = img_temp - np.min(img_temp)
            img_temp /= np.max(img_temp)

            # Clip
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

        idx = 0
        for file in list_IDs_temp:

            # T2-FLAIR channel
            imgFile = os.path.join(
                file, os.path.basename(file) + "_flair.nii.gz")

            img_flair = np.array(nib.load(imgFile).dataobj)
            img_dim = np.shape(img_flair)

            img = np.zeros((img_dim[0], img_dim[1], img_dim[2], self.n_in_channels))

            img[...,0] = img_flair

            if self.n_in_channels > 1:

                # Adding T1 constrast enhanced MRI
                imgFile = os.path.join(
                    file, os.path.basename(file) + "_t1ce.nii.gz")
                img[...,1] = np.array(nib.load(imgFile).dataobj)

                # Adding T1 MRI
                imgFile = os.path.join(
                    file, os.path.basename(file) + "_t1.nii.gz")
                img[...,2] = np.array(nib.load(imgFile).dataobj)

                # Adding T2 MRI
                imgFile = os.path.join(
                    file, os.path.basename(file) + "_t2.nii.gz")
                img[...,3] = np.array(nib.load(imgFile).dataobj)


            # Get mask data
            mskFile = os.path.join(
                file, os.path.basename(file) + "_seg.nii.gz")
            msk = np.array(nib.load(mskFile).dataobj)
            msk[msk > 0] = 1.0   # Combine masks to get whole tumor
            msk = np.expand_dims(msk, -1)

            # Take a crop of the patch_dim size
            img, msk = self.crop_img(img, msk, self.augment)

            img = self.z_normalize_img(img)  # Normalize the image

            # Data augmentation
            if self.augment and (np.random.rand() > 0.5):
                img, msk = self.augment_data(img, msk)

            imgs[idx, ] = img
            msks[idx, ] = msk

            idx += 1

        return imgs, msks
