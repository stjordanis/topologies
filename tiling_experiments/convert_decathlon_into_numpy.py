
import numpy as np
import nibabel as nib   # Needed for reading Nifti files
import os
import glob
from tqdm import tqdm
import h5py

import argparse

parser = argparse.ArgumentParser(
	           description="Convert Medicathlon data "
               "(http://medicaldecathlon.com/) "
               "files to HDF5 data file",
               add_help=True)

parser.add_argument("--data_dir",
					default=os.path.join("/mnt/data/medical/decathlon/",
                                         "Task01_BrainTumour/"),
					help="Root directory for the Medicathlon data files")

parser.add_argument("--save_dir",
					default=os.path.join("/mnt/data/medical/decathlon/",
                                         "Task01_BrainTumour/",
                                         "decathlon_brain"),
					help="Directory to save HDF5 data files")

parser.add_argument("--save_name", default="brain.h5",
					help="HDF5 file name")

parser.add_argument("--split_ratio", type=int, default=0.85,
					help="Split between train and test. Number between 0 and 1")

args = parser.parse_args()

"""
Choose between train and test dataset
"""
dataset = "Train"   # OR Test

def transform_mask_channels(msk):
    """
    Convert mask into separate channels.
    Otherwise, the value is the class ID for the mask.
    """
    newMsk = np.zeros(list(np.shape(msk))+[4])

    for classId in [1, 2, 3]:
         idx, idy, idz = np.where(msk==classId)
         for i in range(len(idx)):
             newMsk[idx[i],idy[i],idz[i],classId] = 1.0

    return newMsk

def normalize_img(img):

    for idx in range(img.shape[3]):
        img[:,:,:,idx] = img[:,:,:,idx] / np.max(img[:,:,:,idx])
    return img

def convert_files(dataDir, saveDir, split_ratio):

    """
    Find filenames for Medicathlon data.
    These are all listed under the subdirectories.
    Directory structure should be something like:
        Task01_BrainTumour/
             imagesTr          # Raw images (training set)
             imagesTs          # Raw images (testing set)
             labelsTr          # Masks (training set)
             labelsTs          # Masks (testing set)

    """
    imgFileNames = glob.glob(os.path.join(dataDir, "imagesTr",
                             "BRATS*.nii.gz"))
    mskFileNames = [filename.replace("imagesTr", "labelsTr")
                    for filename in imgFileNames]

    fileNames = np.array([imgFileNames, mskFileNames])

    image_file = nib.load(fileNames[0,0])
    mask_file = nib.load(fileNames[1,0])

    image_array = image_file.get_data()
    mask_array = mask_file.get_data()

    shapeImage = image_array.shape
    shapeMask = mask_array.shape

    maxshapeImage = (shapeImage[0],shapeImage[1],None,shapeImage[3])
    maxshapeMask = (shapeImage[0], shapeImage[1], None, 4)

    import pathlib
    pathlib.Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    print("Saving HDF5 file to: {}".format(os.path.join(args.save_dir,
                args.save_name)))

    hdfFile = h5py.File(os.path.join(args.save_dir, args.save_name), "w-")
    imgStackTrain = hdfFile.create_dataset("images/train",
                                      data=normalize_img(image_array),
                                      dtype=float,
                                      maxshape=maxshapeImage)
    mskStackTrain = hdfFile.create_dataset("masks/train",
                                      data=transform_mask_channels(mask_array),
                                      dtype=float,
                                      maxshape=maxshapeMask)


    image_file = nib.load(fileNames[0,1])
    mask_file = nib.load(fileNames[1,1])

    image_array = image_file.get_data()
    mask_array = mask_file.get_data()

    shapeImage = image_array.shape
    shapeMask = mask_array.shape

    maxshapeImage = (shapeImage[0],shapeImage[1],None,shapeImage[3])
    maxshapeMask = (shapeImage[0], shapeImage[1], None, 4)

    imgStackTest = hdfFile.create_dataset("images/test",
                                      data=normalize_img(image_array),
                                      dtype=float,
                                      maxshape=maxshapeImage)
    mskStackTest = hdfFile.create_dataset("masks/test",
                                      data=transform_mask_channels(mask_array),
                                      dtype=float,
                                      maxshape=maxshapeMask)

    """
    Go through remaining files in directory and append to stack
    """
    count_train = 1
    count_test = 1

    for idx in tqdm(range(2,fileNames.shape[1])):

        image_file = nib.load(fileNames[0,idx])
        mask_file = nib.load(fileNames[1,idx])

        image_array = image_file.get_data()
        mask_array = mask_file.get_data()

        # Assert that the array shape doesn't change.
        # Otherwise, dstack won't work
        assert(image_array.shape == shapeImage), \
            "File {}: Mismatch shape {}".format(fileNames[0,idx],
            image_array.shape)

        if (np.random.rand() < split_ratio):
            # Train
            count_train += 1
            row = imgStackTrain.shape[2]
            extent = image_array.shape[2]
            imgStackTrain.resize(row+extent, axis=2) # Add new image
            imgStackTrain[:,:,row:(row+extent),:] = \
                        normalize_img(image_array)

            row = mskStackTrain.shape[2]
            extent = mask_array.shape[2]
            mskStackTrain.resize(row+extent, axis=2) # Add new mask
            mskStackTrain[:,:,row:(row+extent),:] = \
                        transform_mask_channels(mask_array)

        else:
            # Test
            count_test += 1
            row = imgStackTest.shape[2]
            extent = image_array.shape[2]
            imgStackTest.resize(row+extent, axis=2) # Add new image
            imgStackTest[:,:,row:(row+extent),:] = \
                        normalize_img(image_array)

            row = mskStackTest.shape[2]
            extent = mask_array.shape[2]
            mskStackTest.resize(row+extent, axis=2) # Add new mask
            mskStackTest[:,:,row:(row+extent),:] = \
                        transform_mask_channels(mask_array)

    imgStackTrain.attrs["lshape"] = np.shape(imgStackTrain)
    mskStackTrain.attrs["lshape"] = np.shape(mskStackTrain)

    imgStackTest.attrs["lshape"] = np.shape(imgStackTest)
    mskStackTest.attrs["lshape"] = np.shape(mskStackTest)

    print("{} images for train, {} for test".format(count_train, count_test))


if __name__ == "__main__":

    print("\n\n\nConverting Medical Decathlon raw "
            "Nifti files to HDF5 data files.")
    print("http://medicaldecathlon.com/")
    print("Looking for decathlon files in: {}".format(args.data_dir))
    print("\nConverting the training files.")
    # Convert the training data
    convert_files(args.data_dir, args.save_dir,args.split_ratio)
