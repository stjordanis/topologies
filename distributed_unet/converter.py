import os
import nibabel as nib
import numpy.ma as ma
import settings_dist
import numpy as np

resize = 128
train_test_split = 0.0  # 0.8  # Just for testing
save_path = settings_dist.PREPROCESS_PATH
save_path = "/home/bduser/data_test/Brats17_TCIA_296_1/data/"

def parse_segments(seg):
    
    msks_parsed = []
    
    for slice in range(seg.shape[-1]):
        curr = seg[:,:,slice]
        GD = ma.masked_not_equal(curr,4).filled(fill_value=0)
        edema = ma.masked_not_equal(curr,2).filled(fill_value=0)
        necrotic = ma.masked_not_equal(curr,1).filled(fill_value=0)
        none = ma.masked_not_equal(curr,0).filled(fill_value=0)
        
        # Stack masks depth-wise
        msks_parsed.append(np.dstack((none,necrotic,edema,GD)))
    
    # Replace all tumorous areas with 1 (previously marked 1, 2 or 4)
    mask = np.asarray(msks_parsed)
    mask[mask > 0] = 1
    
    print("amax",np.amax(mask))
    
    return np.asarray(msks_parsed)
    return mask

def parse_images(img):
    print("image size: ", img.shape)
    slices = []
    for slice in range(img.shape[-1]):
        curr = img[:,:,slice]
        slices.append(curr)
    
    return np.asarray(slices)

def stack_img_slices(mode_track, stack_order):
    
    full_brain = []
    print([(i, len(mode_track[i])) for i in mode_track])
    for slice in range(len(mode_track['t1'])):
        current_slice = []
        for mode in stack_order:
            current_slice.append(mode_track[mode][slice,:,:])
        full_brain.append(np.dstack(current_slice))
        
    stack = np.asarray(full_brain)
    stack = (stack - np.mean(stack))/(np.std(stack))

    return stack

def resize_data(dataset, new_size):
    print("size at resize: ",dataset.shape)
    start_index = (dataset.shape[1] - new_size)/2
    end_index = dataset.shape[1] - start_index

    return np.rot90(dataset[:, start_index:end_index, start_index:end_index :], 3, axes=(1,2))

def save_data(imgs_all, msks_all, split, save_path):
    
    imgs_all = np.asarray(imgs_all)
    msks_all = np.asarray(msks_all)
    
    train_size = int(msks_all.shape[0]*split)

    print("img shape: {}".format(imgs_all.shape))
    print("msk shape: {}\n\n".format(msks_all.shape))

    new_imgs_train = imgs_all[0:train_size,:,:,:]
    new_msks_train = msks_all[0:train_size,:,:,:]
    new_imgs_test = imgs_all[train_size:,:,:,:]
    new_msks_test = msks_all[train_size:,:,:,:]
    
    if os.path.isfile("{}imgs_train.npy".format(save_path)):
             
        imgs_train = np.load("{}imgs_train.npy".format(save_path))
        msks_train = np.load("{}msks_train.npy".format(save_path))
        imgs_test = np.load("{}imgs_test.npy".format(save_path))
        msks_test = np.load("{}msks_test.npy".format(save_path))
            
        np.save("{}imgs_train.npy".format(save_path), np.concatenate((imgs_train,new_imgs_train), axis = 0))
        np.save("{}msks_train.npy".format(save_path), np.concatenate((msks_train,new_msks_train), axis = 0))
        np.save("{}imgs_test.npy".format(save_path), np.concatenate((imgs_test,new_imgs_test), axis = 0))
        np.save("{}msks_test.npy".format(save_path), np.concatenate((msks_test,new_imgs_test), axis = 0))

    else:
        
        np.save("{}imgs_train.npy".format(save_path), new_imgs_train)
        np.save("{}msks_train.npy".format(save_path), new_msks_train)
        np.save("{}imgs_test.npy".format(save_path), new_imgs_test)
        np.save("{}msks_test.npy".format(save_path), new_msks_test)
        
    
root_dir = '/home/bduser/data_test/Brats17_TCIA_296_1'
imgs_all = []
msks_all = []
dummy = 0
for subdir, dir, files in os.walk(root_dir):

    # Ensure all necessary files are present
    file_root = subdir.split('/')[-1] + "_"
    extension = ".nii.gz"
    img_modes = ["t1","t2","flair","t1ce"]
    #img_modes = ["flair"]
    need_file = [file_root + mode + extension for mode in img_modes]
    all_there = [(reqd in files) for reqd in need_file]
    if all(all_there) and dummy < 1:
        
        mode_track = {mode:[] for mode in img_modes}
        for file in files:

            if file.endswith('seg.nii.gz'):
                path = os.path.join(subdir,file)
                msk = np.array(nib.load(path).dataobj)
                parsed = resize_data(parse_segments(msk), resize)
                msks_all.extend(parsed)

            if file.endswith('t1.nii.gz'):
                path = os.path.join(subdir,file)
                img = np.array(nib.load(path).dataobj)
                mode_track['t1'] = resize_data(parse_images(img), resize)

            if file.endswith('t2.nii.gz'):
                path = os.path.join(subdir,file)
                img = np.array(nib.load(path).dataobj)
                mode_track['t2'] = resize_data(parse_images(img), resize)

            if file.endswith('t1ce.nii.gz'):
                path = os.path.join(subdir,file)
                img = np.array(nib.load(path).dataobj)
                mode_track['t1ce'] = resize_data(parse_images(img), resize)

            if file.endswith('flair.nii.gz'):
                path = os.path.join(subdir,file)
                img = np.array(nib.load(path).dataobj)
                mode_track['flair'] = resize_data(parse_images(img), resize)

        dummy += 1
        imgs_all.extend(np.asarray(stack_img_slices(mode_track,img_modes)))
        print("dummy: {0}, msks_all: {1}, imgs_all: {2}".format(dummy, len(msks_all), len(imgs_all)))

    if (dummy == 1) & (dummy != 0):
        print("Total brains: {}".format(dummy))
        save_data(imgs_all, msks_all, train_test_split, save_path)
        imgs_all = []
        msks_all = []
        print("Saved checkpoint")
