
import numpy as np
import nibabel as nib   # Needed for reading Nifti files
import os
import glob
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(
	           description="Convert Medicathlon data "
               "files to Numpy data file",
               add_help=True)

parser.add_argument("--data_dir",
					default=os.path.join("/mnt/data/medical/decathlon/",
                                         "Task01_BrainTumour/"),
					help="Root directory for the Medicathlon data files")

parser.add_argument("--save_dir",
					default=os.path.join(".", "decathlon_brain"),
					help="Directory to save Numpy data files")

args = parser.parse_args()

"""
Choose between train and test dataset
"""
dataset = "Train"   # OR Test

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

    image_array = image_file.get_data()
    mask_array = mask_file.get_data()

    shapeImage = image_array.shape
    shapeMask = mask_array.shape
    okFiles = 1

    imgStack = image_array
    mskStack = mask_array

    """
    Go through remaining files in directory and append to stack
    """
    for idx in tqdm(range(1,fileNames.shape[1])):

        image_file = nib.load(fileNames[0,idx])
        mask_file = nib.load(fileNames[1,idx])

        image_array = image_file.get_data()
        mask_array = mask_file.get_data()

        # Assert that the array shape doesn't change.
        # Otherwise, dstack won't work
        assert(image_array.shape == shapeImage), \
            "File {}: Mismatch shape {}".format(fileNames[0,idx],
            image_array.shape)

        imgStack = np.dstack((imgStack, image_array))
        mskStack = np.dstack((mskStack, mask_array))


    """
    Store the different masks as different channels rather
    than as class ids.
    """
    newMaskStack = np.zeros(list(mskStack.shape) + [4])
    for mask in [1, 2, 3]:
        idx, idy, idz = np.where(mskStack==mask)
        for i in range(len(idx)):
            newMaskStack[idx[i],idy[i],idz[i],mask] = 1.0

    """
    Save the Numpy Array to file
    """

    import pathlib
    pathlib.Path(saveDir).mkdir(parents=True, exist_ok=False)

    print("Saving data arrays to Numpy. Please wait...")
    np.save("{}/imgs_{}.npy".format(saveDir, appendSave), imgStack)
    print("Saved file ".format("{}/imgs_{}.npy".format(saveDir, appendSave)))
    np.save("{}/msks_{}.npy".format(saveDir, appendSave), newMaskStack)
    print("Saved file ".format("{}/msks_{}.npy".format(saveDir, appendSave)))


if __name__ == "__main__":

    print("\n\n\nConverting Medical Decathlon raw files to Numpy data files.")
    print("http://medicaldecathlon.com/")
    print("Looking for decathlon files in: {}".format(args.data_dir))
    print("Saving numpy files in: {}".format(args.save_dir))
    print("\nConverting the training files.")
    # Convert the training data
    convert_files(args.data_dir, args.save_dir, True)

    print("Converting the testing files.")
    # Convert the testing data
    convert_files(args.data_dir, args.save_dir, False)
