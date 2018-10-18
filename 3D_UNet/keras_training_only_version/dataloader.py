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
                 augment=False):   # Augment images
        """
        Initialization
        """
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.indexes = np.arange(len(list_IDs))
        self.n_in_channels = n_in_channels
        self.n_out_channels = n_out_channels
        self.shuffle = shuffle
        self.augment = augment

    def __len__(self):
        """
        The number of batches per epoch
        """
        return len(self.list_IDs) // self.batch_size

    def __getitem__(self, index):
        """
        Generate one batch of data
        """
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        If shuffle is true, then it will shuffle the training set
        after every epoch.
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
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

            ratio_crop = 0.10  # Crop up this this % of pixels for offset
            # Number of pixels to offset crop in this dimension
            offset = int(np.floor(start*ratio_crop))

            if randomize:
                start += np.random.choice(range(-offset, offset))
                if ((start + cropLen) > imgLen):  # Don't fall off the image
                    start = (imgLen-cropLen)//2

            slices.append(slice(start, start+cropLen))

        return img[tuple(slices)], msk[tuple(slices)]

    def augment_data(self, img, msk):
        """
        Data augmentation
        Flip image and mask. Rotate image and mask.
        """

        if np.random.rand() > 0.5:
            # Random 0,1,2 (axes to flip)
            ax = np.random.choice(np.arange(len(self.dim)))
            img = np.flip(img, ax)
            msk = np.flip(msk, ax)

        elif np.random.rand() > 0.5:
            rot = np.random.choice([1, 2, 3])  # 90, 180, or 270 degrees
            img = np.rot90(img, rot)
            msk = np.rot90(msk, rot)

        return img, msk

    def z_normalize_img(self, img):
        """
        Normalize the image so that the mean value for each image
        is 0 and the standard deviation is 1.
        """
        return (img - np.mean(img)) / np.std(img)

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

            imgFile = os.path.join(
                file, os.path.basename(file) + "_flair.nii.gz")
            mskFile = os.path.join(
                file, os.path.basename(file) + "_seg.nii.gz")

            img = np.array(nib.load(imgFile).dataobj)

            msk = np.array(nib.load(mskFile).dataobj)
            msk[msk > 0] = 1.0   # Combine masks to get whole tumor

            # Take a crop of the patch_dim size
            img, msk = self.crop_img(img, msk, self.augment)

            img = self.z_normalize_img(img)  # Normalize the image

            # Data augmentation
            if self.augment and (np.random.rand() > 0.5):
                img, msk = self.augment_data(img, msk)

            imgs[idx,] = np.expand_dims(img, -1)
            msks[idx,] = np.expand_dims(msk, -1)

            idx += 1

        return imgs, msks
