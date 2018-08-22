
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
					default=os.path.join(".", "decathlon_brain"),
					help="Directory to save HDF5 data files")

parser.add_argument("--save_name", default="brain.h5",
					help="HDF5 file name")

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

def convert_files(dataDir, saveDir, isTrain):

    if isTrain:  # Work on training set
        appendDir = "Tr"
        appendSave = "train"
    else:
        appendDir = "Ts"  # Work on testing set
        appendSave = "test"

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
    imgFileNames = glob.glob(os.path.join(dataDir,
                             "images{}".format(appendDir),
                             "BRATS*.nii.gz"))
    mskFileNames = [filename.replace("images{}".format(appendDir),
                    "labels{}".format(appendDir))
                    for filename in imgFileNames]

    fileNames = np.array([imgFileNames, mskFileNames])

    image_file = nib.load(fileNames[0,0])
    mask_file = nib.load(fileNames[1,0])

    image_array = normalize_img(image_file.get_data())
    mask_array = mask_file.get_data()

    shapeImage = image_array.shape
    shapeMask = mask_array.shape
    okFiles = 1

    maxshapeImage = (shapeImage[0],shapeImage[1],None,shapeImage[3])
    maxshapeMask = (shapeImage[0], shapeImage[1], None, 4)

    import pathlib
    pathlib.Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    print("Saving HDF5 file to: {}".format(os.path.join(args.save_dir,
                appendSave,
                args.save_name)))

    hdfFile = h5py.File(os.path.join(args.save_dir, args.save_name), "w-")
    imgStack = hdfFile.create_dataset("images",
                                      data=image_array,
                                      dtype=float,
                                      maxshape=maxshapeImage)
    mskStack = hdfFile.create_dataset("masks",
                                      data=transform_mask_channels(mask_array),
                                      dtype=float,
                                      maxshape=maxshapeMask)

    """
    Go through remaining files in directory and append to stack
    """
    for idx in tqdm(range(1,fileNames.shape[1])):

        image_file = nib.load(fileNames[0,idx])
        mask_file = nib.load(fileNames[1,idx])

        image_array = normalize_img(image_file.get_data())
        mask_array = mask_file.get_data()

        # Assert that the array shape doesn't change.
        # Otherwise, dstack won't work
        assert(image_array.shape == shapeImage), \
            "File {}: Mismatch shape {}".format(fileNames[0,idx],
            image_array.shape)

        row = imgStack.shape[2]
        extent = image_array.shape[2]
        imgStack.resize(row+extent, axis=2) # Add new image
        imgStack[:,:,row:(row+extent),:] = image_array

        ow = mskStack.shape[2]
        extent = mask_array.shape[2]
        mskStack.resize(row+extent, axis=2) # Add new mask
        mskStack[:,:,row:(row+extent),:] = transform_mask_channels(mask_array)

    imgStack.attrs["lshape"] = np.shape(imgStack)
    imgStack.attrs["type"] = appendSave
    mskStack.attrs["lshape"] = np.shape(mskStack)
    mskStack.attrs["type"] = appendSave

if __name__ == "__main__":

    print("\n\n\nConverting Medical Decathlon raw "
            "Nifti files to HDF5 data files.")
    print("http://medicaldecathlon.com/")
    print("Looking for decathlon files in: {}".format(args.data_dir))
    print("\nConverting the training files.")
    # Convert the training data
    convert_files(args.data_dir, args.save_dir, True)

    print("Converting the testing files.")
    # Convert the testing data
    convert_files(args.data_dir, args.save_dir, False)
