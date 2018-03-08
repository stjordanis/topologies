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

import os.path
import numpy as np
import tensorflow as tf

def dice_coef(target, prediction, axis=(1, 2, 3), smooth=1e-5):
	'''
	Sorenson Dice
	'''
	intersection = tf.reduce_sum(prediction * target, axis=axis)
	p = tf.reduce_sum(prediction, axis=axis)
	t = tf.reduce_sum(target, axis=axis)
	dice = (2. * intersection + smooth) / (t + p + smooth)

	return tf.reduce_mean(dice)

def dice_coef_loss(target, prediction, axis=(1,2,3), smooth=1e-5):
	'''
	Sorenson Dice loss
	Using -log(Dice) as the loss since it is better behaved.
	Also, the log allows avoidance of the division which
	can help prevent underflow when the numbers are very small.
	'''
	intersection = tf.reduce_sum(prediction * target, axis=axis)
	p = tf.reduce_sum(prediction, axis=axis)
	t = tf.reduce_sum(target, axis=axis)
	numerator = tf.reduce_mean(2. * intersection + smooth)
	denominator = tf.reduce_mean(t + p + smooth)
	dice_loss = -tf.log(numerator) + tf.log(denominator)

	return dice_loss

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

def define_model(input_img, use_upsampling=False, learning_rate=0.001, n_cl_out=1, dropout=0.2, print_summary = False):

	# [b,h,w,d,c] = input_img.shape
	# if not is_power_of_2(h) or  \
	# 	not is_power_of_2(w) or \
	# 	not is_power_of_2(d):
	# 	print("ERROR: Image dimension lengths must be a power of 2. e.g. 16x256x32")

	# inputs = tf.keras.layers.Input(shape=(h,w,d,c), name="Input_Image")

	# Set keras learning phase to train
	tf.keras.backend.set_learning_phase(True)

	# Don't initialize variables on the fly
	tf.keras.backend.manual_variable_initialization(False)

	inputs = tf.keras.layers.Input(tensor=input_img, name="Input_Image")

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

	if use_upsampling:
		up4 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling3D(name="up4", size=(2, 2, 2))(conv4), conv3], axis=concat_axis)
	else:
		up4 = tf.keras.layers.concatenate([tf.keras.layers.Conv3DTranspose(name="transConv4", filters=512, data_format=data_format,
						   kernel_size=(2, 2, 2), strides=(2, 2, 2), padding="same")(conv4), conv3], axis=concat_axis)

	conv5 = tf.keras.layers.Conv3D(name="conv5a", filters=256, **params)(up4)
	conv5 = tf.keras.layers.Conv3D(name="conv5b", filters=256, **params)(conv5)

	if use_upsampling:
		up5 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling3D(name="up5", size=(2, 2, 2))(conv5), conv2], axis=concat_axis)
	else:
		up5 = tf.keras.layers.concatenate([tf.keras.layers.Conv3DTranspose(name="transConv5", filters=256, data_format=data_format,
						   kernel_size=(2, 2, 2), strides=(2, 2, 2), padding="same")(conv5), conv2], axis=concat_axis)

	conv6 = tf.keras.layers.Conv3D(name="conv6a", filters=128, **params)(up5)
	conv6 = tf.keras.layers.Conv3D(name="conv6b", filters=128, **params)(conv6)

	if use_upsampling:
		up6 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling3D(name="up6", size=(2, 2, 2))(conv6), conv1], axis=concat_axis)
	else:
		up6 = tf.keras.layers.concatenate([tf.keras.layers.Conv3DTranspose(name="transConv6", filters=128, data_format=data_format,
						   kernel_size=(2, 2, 2), strides=(2, 2, 2), padding="same")(conv6), conv1], axis=concat_axis)

	conv7 = tf.keras.layers.Conv3D(name="conv7a", filters=128, **params)(up6)
	conv7 = tf.keras.layers.Conv3D(name="conv7b", filters=128, **params)(conv7)
	pred = tf.keras.layers.Conv3D(name="Prediction_Mask", filters=n_cl_out, kernel_size=(1, 1, 1),
					data_format=data_format, activation="sigmoid")(conv7)

	model = tf.keras.models.Model(inputs=[inputs], outputs=[pred])

	if print_summary:
		model.summary()

	# optimizer = tf.train.AdamOptimizer(learning_rate)
	# model.compile(optimizer=optimizer, loss=dice_coef_loss, metrics=[dice_coef])

	return pred #model


def sensitivity(target, prediction, axis=(1,2,3), smooth = 1e-5 ):

	intersection = tf.reduce_sum(prediction * target, axis=axis)
	coef = (intersection + smooth) / (tf.reduce_sum(prediction, axis=axis) + smooth)
	return tf.reduce_mean(coef)

def specificity(target, prediction, axis=(1,2,3), smooth = 1e-5 ):

	intersection = tf.reduce_sum(prediction * target, axis=axis)
	coef = (intersection + smooth) / (tf.reduce_sum(prediction, axis=axis) + smooth)
	return tf.reduce_mean(coef)
