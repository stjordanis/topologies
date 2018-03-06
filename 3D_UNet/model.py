import os.path
import numpy as np
import tensorflow as tf

def dice_coef(prediction, target, axis=(1, 2, 3), smooth=1e-5):

	intersection = tf.reduce_sum(prediction * target, axis=axis)
	p = tf.reduce_sum(prediction, axis=axis)
	t = tf.reduce_sum(target, axis=axis)
	dice = (2. * intersection + smooth) / (t + p + smooth)

	return tf.reduce_mean(dice)

def dice_coef_loss(prediction, target, axis=(1,2,3), smooth=1e-5):
	return -tf.log(dice_coef(prediction,target, axis, smooth))

CHANNEL_LAST = True
if CHANNEL_LAST:
	concat_axis = -1
	data_format = 'channels_last'

else:
	concat_axis = 1
	data_format = 'channels_first'

tf.keras.backend.set_image_data_format(data_format)

def is_power_of_2(num):
	'''
	Check if number is a power of 2
	If not, then the U-Net model might not compile
	because the feature maps round down on odd lengths.
	For example, if the image is 10x10x10. The first MaxPooling3D
	will reduce it to 5x5x5. The second MaxPooling3D will make it 2x2x2.
	However, the UpSampling3D will take the 2x2x2 and make it 4x4x4.
	That means you try to concatenate a 5x5x5 on the encoder with a
	4x4x4 on the decoder (which gives an error).
	'''
	return ((num & (num - 1)) == 0) and num > 0

def define_model(input_img, use_upsampling=False, n_cl_out=1, dropout=0.2, print_summary = False):

	[b,h,w,d,c] = input_img.shape
	if not is_power_of_2(h) or  \
		not is_power_of_2(w) or \
		not is_power_of_2(d):
		print("ERROR: Image dimension lengths must be a power of 2. e.g. 16x256x32")

	inputs = tf.keras.layers.Input(shape=(h,w,d,c), name="Input_Image")

	params = dict(kernel_size=(3, 3, 3), activation="relu",
				  padding="same", data_format=data_format,
				  kernel_initializer="he_uniform")

	conv1 = tf.keras.layers.Conv3D(name="conv1a", filters=32, **params)(inputs)
	conv1 = tf.keras.layers.Conv3D(name="conv1b", filters=64, **params)(conv1)
	pool1 = tf.keras.layers.MaxPooling3D(name="pool1", pool_size=(2, 2, 2))(conv1)

	conv2 = tf.keras.layers.Conv3D(name="conv2a", filters=64, **params)(pool1)
	conv2 = tf.keras.layers.Conv3D(name="conv2b", filters=128, **params)(conv2)
	pool2 = tf.keras.layers.MaxPooling3D(name="pool2", pool_size=(2, 2, 2))(conv2)

	conv3 = tf.keras.layers.Conv3D(name="conv3a", filters=128, **params)(pool2)
	conv3 = tf.keras.layers.Dropout(dropout)(conv3) ### Trying dropout layers earlier on, as indicated in the paper
	conv3 = tf.keras.layers.Conv3D(name="conv3b", filters=256, **params)(conv3)
	pool3 = tf.keras.layers.MaxPooling3D(name="pool3", pool_size=(2, 2, 2))(conv3)

	conv4 = tf.keras.layers.Conv3D(name="conv4a", filters=256, **params)(pool3)
	conv4 = tf.keras.layers.Dropout(dropout)(conv4) ### Trying dropout layers earlier on, as indicated in the paper
	conv4 = tf.keras.layers.Conv3D(name="conv4b", filters=512, **params)(conv4)

	up4 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling3D(name="up4", size=(2, 2, 2))(conv4), conv3], axis=concat_axis)

	conv5 = tf.keras.layers.Conv3D(name="conv5a", filters=256, **params)(up4)
	conv5 = tf.keras.layers.Conv3D(name="conv5b", filters=256, **params)(conv5)

	up5 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling3D(name="up5", size=(2, 2, 2))(conv5), conv2], axis=concat_axis)

	conv6 = tf.keras.layers.Conv3D(name="conv6a", filters=128, **params)(up5)
	conv6 = tf.keras.layers.Conv3D(name="conv6b", filters=128, **params)(conv6)

	up6 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling3D(name="up6", size=(2, 2, 2))(conv6), conv1], axis=concat_axis)

	conv7 = tf.keras.layers.Conv3D(name="conv7a", filters=128, **params)(up6)
	conv7 = tf.keras.layers.Conv3D(name="conv7b", filters=128, **params)(conv7)
	pred = tf.keras.layers.Conv3D(name="Prediction_Mask", filters=n_cl_out, kernel_size=(1, 1, 1),
					data_format=data_format, activation="sigmoid")(conv7)

	model = tf.keras.models.Model(inputs=[inputs], outputs=[pred])

	if print_summary:
		print (model.summary())

	optimizer = tf.train.AdamOptimizer(0.001)
	model.compile(optimizer=optimizer, loss=dice_coef_loss, metrics=[dice_coef])

	return model


def sensitivity(y_true, y_pred, smooth = 1. ):

	intersection = tf.reduce_sum(y_true * y_pred)
	coef = (intersection + smooth) / (tf.reduce_sum(y_true) + smooth)
	return tf.reduce_mean(coef)

def specificity(y_true, y_pred, smooth = 1. ):

	intersection = tf.reduce_sum(y_true * y_pred)
	coef = (intersection + smooth) / (tf.reduce_sum(y_pred) + smooth)
	return tf.reduce_mean(coef)
