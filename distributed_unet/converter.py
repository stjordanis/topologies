import os
import nibabel as nib
import numpy.ma as ma
import settings_dist
import numpy as np
import matplotlib.pyplot as plt

im_num = 85
channel = 2
resize = 128
train_test_split = 0.8
save_path = settings_dist.OUT_TEST_PATH

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
	
	return np.asarray(msks_parsed)

def parse_images(img):
	
	slices = []
	for slice in range(img.shape[-1]):
		curr = img[:,:,slice]
		slices.append(curr)
	
	return np.asarray(slices)

def stack_img_slices(mode_track, stack_order):
	
	full_brain = []
	for slice in range(len(mode_track['t1'])):
		current_slice = []
		for mode in stack_order:
			current_slice.append(mode_track[mode][slice,:,:])
		full_brain.append(np.dstack(current_slice))

	return np.asarray(full_brain)

def resize_data(dataset, new_size):
	print(dataset.shape)
	start_index = (dataset.shape[1] - new_size)/2
	end_index = dataset.shape[1] - start_index

	return dataset[:, start_index:end_index, start_index:end_index :]

def save_data(imgs_all, msks_all, split, save_path):
	
	imgs_all = np.asarray(imgs_all, dtype=np.float64)
	msks_all = np.asarray(msks_all)
	
	train_size = int(msks_all.shape[0]*split)

	print("img shape: {}".format(imgs_all.shape))
	print("msk shape: {}\n\n".format(msks_all.shape))

	new_imgs_train = imgs_all[0:train_size,:,:,:]
	new_msks_train = msks_all[0:train_size,:,:,:]
	new_imgs_test = imgs_all[train_size:,:,:,:]
	new_msks_test = msks_all[train_size:,:,:,:]
	
	if os.path.isfile("{}NEW-imgs_train.npy".format(save_path)):
		
		
		imgs_train = np.load('{}NEW-imgs_train.npy'.format(save_path))
		msks_train = np.load('{}NEW-msks_train.npy'.format(save_path))
		imgs_test = np.load('{}NEW-imgs_test.npy'.format(save_path))
		msks_test = np.load('{}NEW-msks_test.npy'.format(save_path))
	
		print(imgs_train.shape)
		print(new_imgs_train.shape)
		print(np.concatenate((imgs_train,new_imgs_train), axis = 0).shape)
		
		np.save("{}NEW-imgs_train.npy", np.concatenate((imgs_train,new_imgs_train), axis = 0))
		np.save("{}NEW-msks_train.npy", np.concatenate((msks_train,new_msks_train), axis = 0))
		np.save("{}NEW-imgs_test.npy", np.concatenate((imgs_test,new_imgs_test), axis = 0))
		np.save("{}NEW-msks_test.npy", np.concatenate((imgs_test,new_imgs_test), axis = 0))
	
	else:
		
		np.save("{}NEW-imgs_train.npy".format(save_path), new_imgs_train)
		np.save("{}NEW-msks_train.npy".format(save_path), new_msks_train)
		np.save("{}NEW-imgs_test.npy".format(save_path), new_imgs_test)
		np.save("{}NEW-msks_test.npy".format(save_path), new_imgs_test)
		
	
root_dir = '/home/bduser/data_test/MICCAI_BraTS17_Data_Training'
imgs_all = []
msks_all = []
dummy = 1
for subdir, dir, files in os.walk(root_dir):

	# Ensure all necessary files are present
	file_root = subdir.split('/')[-1] + "_"
	extension = ".nii.gz"
	img_modes = ["t1","t2","flair","t1ce"]
	need_file = [file_root + mode + extension for mode in img_modes]
	all_there = [(reqd in files) for reqd in need_file]
	if all(all_there):
		
		mode_track = {mode:[] for mode in img_modes}
		for file in files:
		
			if file.startswith('Brats17_2013') & file.endswith('seg.nii.gz'):
				path = os.path.join(subdir,file)
				msk = np.array(nib.load(path).dataobj)
				parsed = parse_segments(msk)
				msks_all.extend(parsed)
				print("len msksall",len(msks_all))
				dummy += 1
			elif file.startswith('Brats17_2013'):
				path = os.path.join(subdir,file)
				if file.endswith('t1.nii.gz'):
					img = np.array(nib.load(path).dataobj)
					mode_track['t1'] = parse_images(img)

				if file.endswith('t2.nii.gz'):
					img = np.array(nib.load(path).dataobj)
					mode_track['t2'] = parse_images(img)

				if file.endswith('t1ce.nii.gz'):
					img = np.array(nib.load(path).dataobj)
					mode_track['t1ce'] = parse_images(img)

				if file.endswith('flair.nii.gz'):
					img = np.array(nib.load(path).dataobj)
					mode_track['flair'] = parse_images(img)
		
		imgs_all.extend(np.asarray(stack_img_slices(mode_track,img_modes)))
	
	#dummy += 1
	if dummy%3 == 0:
		print("Total brains: {}".format(dummy))
		save_data(imgs_all, msks_all, train_test_split, save_path)
		imgs_all = []
		msks_all = []
		print("Saved checkpoint")

# Resize images to 128*128 (default)
#if resize != 0:
#    imgs_all = resize_data(imgs_all, resize)
#    msks_all = resize_data(msks_all, resize)

#np.save("msks_all.npy", msks_all)


# Save training and test sets according to train_test_split ratio


#np.save("{}NEW-msks_train.npy".format(save_path),msks_all[0:train_size,:,:,:])
#np.save("{}NEW-msks_test.npy".format(save_path),msks_all[train_size:,:,:,:])
#np.save("{}NEW-imgs_train.npy".format(save_path),imgs_all[0:train_size,:,:,:])
#np.save("{}NEW-imgs_test.npy".format(save_path),msks_all[train_size:,:,:,:])

#image = imgs_all[im_num]
#mask = msks_all[im_num][:,:,channel]
#f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
#ax1.imshow(image, cmap='gray')
#ax2.imshow(mask)
#plt.show()