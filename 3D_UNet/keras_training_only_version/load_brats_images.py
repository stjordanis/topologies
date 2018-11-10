import os
import random
import nibabel as nib
import numpy as np
from tqdm import tqdm

def get_file_list(data_path="../../../../data/"):

    fileList = []
    for subdir, dir, files in os.walk(data_path):
        # Make sure directory has data
        if os.path.isfile(os.path.join(subdir, os.path.basename(subdir) + "_flair.nii.gz")):
            fileList.append(subdir)

    random.Random(816).shuffle(fileList)
    n = len(fileList)
    train_test_split = 0.85  # 85% train test split
    train_length = int(train_test_split*n)
    trainList = fileList[:train_length]
    testList = fileList[train_length:]

    return trainList, testList


def get_batch(fileList, batch_size=8):

    patch_size = 128

    def crop_center(img,cropx=patch_size,cropy=patch_size,cropz=patch_size):
        x,y,z = img.shape
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)
        startz = z//2-(cropz//2)
        return img[startx:startx+cropx,starty:starty+cropy,startz:startz+cropz]

    random.shuffle(fileList)
    files = fileList[:batch_size]

    imgs = np.zeros((batch_size, patch_size, patch_size, patch_size, 1))
    msks = np.zeros((batch_size, patch_size, patch_size, patch_size, 1))

    idx = 0
    for file in tqdm(files):

        imgFile = os.path.join(file, os.path.basename(file) + "_flair.nii.gz")
        mskFile = os.path.join(file, os.path.basename(file) + "_seg.nii.gz")

        img = np.array(nib.load(imgFile).dataobj)

        img = crop_center(img)
        img = (img - np.mean(img)) / np.std(img)  # z normalize image

        msk = np.array(nib.load(mskFile).dataobj)
        msk[msk > 0] = 1.0   # Combine masks to get whole tumor
        msk = crop_center(msk)

        imgs[idx,:, :, :, 0] = img
        msks[idx,:, :, :, 0] = msk

        idx += 1

    return imgs, msks

def get_all(fileList):

    patch_size = 128

    def crop_center(img,cropx=patch_size,cropy=patch_size,cropz=patch_size):
        x,y,z = img.shape
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)
        startz = z//2-(cropz//2)
        return img[startx:startx+cropx,starty:starty+cropy,startz:startz+cropz]

    imgs = np.zeros((len(fileList), patch_size, patch_size, patch_size, 1))
    msks = np.zeros((len(fileList), patch_size, patch_size, patch_size, 1))

    idx = 0
    for file in tqdm(fileList):

        imgFile = os.path.join(file, os.path.basename(file) + "_flair.nii.gz")
        mskFile = os.path.join(file, os.path.basename(file) + "_seg.nii.gz")

        img = np.array(nib.load(imgFile).dataobj)

        img = crop_center(img)
        img = (img - np.mean(img)) / np.std(img)  # z normalize image

        msk = np.array(nib.load(mskFile).dataobj)
        msk[msk > 0] = 1.0   # Combine masks to get whole tumor
        msk = crop_center(msk)

        imgs[idx,:, :, :, 0] = img
        msks[idx,:, :, :, 0] = msk

        idx += 1

    return imgs, msks

trainList, testList = get_file_list()
#imgs, msks = get_batch(trainList,8)

imgs, msks = get_all(testList)
np.save("imgs_test_3d.npy", imgs)
np.save("msks_test_3d.npy", msks)

print("Saved imgs and masks from test dataset to Numpy data files")
#print(imgs.shape)
#print(msks.shape)
