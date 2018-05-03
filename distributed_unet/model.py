import os.path
import tensorflow as tf
from tensorflow import keras as K

import settings_dist

def dice_coef(y_true, y_pred, smooth = 1. ):

	#y_true_f = tf.convert_to_tensor(y_true)

	intersection = tf.reduce_sum(y_true * y_pred)
	coef = (tf.constant(2.) * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)
	return coef


def dice_coef_loss(y_true, y_pred, smooth = 1.):

	#y_true_f = tf.convert_to_tensor(y_true)

	intersection = tf.reduce_sum(y_true * y_pred)

	loss = -tf.log(tf.constant(2.) * intersection + smooth) + \
		tf.log((tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth))
	return loss

def sensitivity(y_true, y_pred, smooth = 1. ):

	intersection = tf.reduce_sum(y_true * y_pred)
	coef = (intersection + smooth) / (tf.reduce_sum(y_true) + smooth)
	return coef

def specificity(y_true, y_pred, smooth = 1. ):

	intersection = tf.reduce_sum(y_true * y_pred)
	coef = (intersection + smooth) / (tf.reduce_sum(y_pred) + smooth)
	return coef


CHANNEL_LAST = True
if CHANNEL_LAST:
	concat_axis = -1
	data_format = 'channels_last'

else:
	concat_axis = 1
	data_format = 'channels_first'

K.backend.set_image_data_format(data_format)

def define_model(FLAGS, input_shape, output_shape, num_replicas):

	# # Set keras learning phase to train
	# K.backend.set_learning_phase(True)
	#
	# # Don't initialize variables on the fly
	# K.backend.manual_variable_initialization(False)

	n_cl_out = 1 # Number of output classes
	dropout = 0.2   # Percentage of dropout for network layers


	imgs = tf.placeholder(tf.float32, shape=([None] + list(input_shape)))

	msks = tf.placeholder(tf.float32, shape=([None] + list(output_shape)))

	# Decay learning rate from initial_learn_rate to initial_learn_rate*fraction in decay_steps global steps
	if FLAGS.const_learningrate:
		learning_rate = tf.convert_to_tensor(FLAGS.learning_rate, dtype=tf.float32)
	else:
		learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,
			global_step, FLAGS.decay_steps, FLAGS.lr_fraction, staircase=False)


	inputs = K.layers.Input(tensor=imgs, name='Images')

	params = dict(kernel_size=(3, 3), activation='relu',
				  padding='same', data_format=data_format,
				  kernel_initializer='he_uniform') #RandomUniform(minval=-0.01, maxval=0.01, seed=816))

	conv1 = K.layers.Conv2D(name='conv1a', filters=32, **params)(inputs)
	conv1 = K.layers.Conv2D(name='conv1b', filters=32, **params)(conv1)
	pool1 = K.layers.MaxPooling2D(name='pool1', pool_size=(2, 2))(conv1)

	conv2 = K.layers.Conv2D(name='conv2a', filters=64, **params)(pool1)
	conv2 = K.layers.Conv2D(name='conv2b', filters=64, **params)(conv2)
	pool2 = K.layers.MaxPooling2D(name='pool2', pool_size=(2, 2))(conv2)

	conv3 = K.layers.Conv2D(name='conv3a', filters=128, **params)(pool2)
	conv3 = K.layers.Dropout(dropout)(conv3) ### Trying dropout layers earlier on, as indicated in the paper
	conv3 = K.layers.Conv2D(name='conv3b', filters=128, **params)(conv3)

	pool3 = K.layers.MaxPooling2D(name='pool3', pool_size=(2, 2))(conv3)

	conv4 = K.layers.Conv2D(name='conv4a', filters=256, **params)(pool3)
	conv4 = K.layers.Dropout(dropout)(conv4) ### Trying dropout layers earlier on, as indicated in the paper
	conv4 = K.layers.Conv2D(name='conv4b', filters=256, **params)(conv4)

	pool4 = K.layers.MaxPooling2D(name='pool4', pool_size=(2, 2))(conv4)

	conv5 = K.layers.Conv2D(name='conv5a', filters=512, **params)(pool4)
	conv5 = K.layers.Conv2D(name='conv5b', filters=512, **params)(conv5)

	if FLAGS.use_upsampling:
		up6 = K.layers.concatenate([K.layers.UpSampling2D(name='up6', size=(2, 2))(conv5), conv4], axis=concat_axis)
	else:
		up6 = K.layers.concatenate([K.layers.Conv2DTranspose(name='transConv6', filters=256, data_format=data_format,
			               kernel_size=(2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=concat_axis)

	conv6 = K.layers.Conv2D(name='conv6a', filters=256, **params)(up6)
	conv6 = K.layers.Conv2D(name='conv6b', filters=256, **params)(conv6)

	if FLAGS.use_upsampling:
		up7 = K.layers.concatenate([K.layers.UpSampling2D(name='up7', size=(2, 2))(conv6), conv3], axis=concat_axis)
	else:
		up7 = K.layers.concatenate([K.layers.Conv2DTranspose(name='transConv7', filters=128, data_format=data_format,
			               kernel_size=(2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=concat_axis)

	conv7 = K.layers.Conv2D(name='conv7a', filters=128, **params)(up7)
	conv7 = K.layers.Conv2D(name='conv7b', filters=128, **params)(conv7)

	if FLAGS.use_upsampling:
		up8 = K.layers.concatenate([K.layers.UpSampling2D(name='up8', size=(2, 2))(conv7), conv2], axis=concat_axis)
	else:
		up8 = K.layers.concatenate([K.layers.Conv2DTranspose(name='transConv8', filters=64, data_format=data_format,
			               kernel_size=(2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=concat_axis)


	conv8 = K.layers.Conv2D(name='conv8a', filters=64, **params)(up8)
	conv8 = K.layers.Conv2D(name='conv8b', filters=64, **params)(conv8)

	if FLAGS.use_upsampling:
		up9 = K.layers.concatenate([K.layers.UpSampling2D(name='up9', size=(2, 2))(conv8), conv1], axis=concat_axis)
	else:
		up9 = K.layers.concatenate([K.layers.Conv2DTranspose(name='transConv9', filters=32, data_format=data_format,
			               kernel_size=(2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=concat_axis)


	conv9 = K.layers.Conv2D(name='conv9a', filters=32, **params)(up9)
	conv9 = K.layers.Conv2D(name='conv9b', filters=32, **params)(conv9)

	predictionMask = K.layers.Conv2D(name='Mask', filters=n_cl_out, kernel_size=(1, 1),
					data_format=data_format, activation='sigmoid')(conv9)

	"""
	Define the variables, losses, and metrics
	We'll return these as a dictionary called "model"
	"""
	model = {}
	model["input"] = imgs
	model["label"] = msks
	model["output"] = predictionMask
	model["loss"] = dice_coef_loss(msks, predictionMask)
	model["metric_dice"] = dice_coef(msks, predictionMask)
	model["metric_sensitivity"] = sensitivity(msks, predictionMask)
	model["metric_specificity"] = specificity(msks, predictionMask)

	model["global_step"] = tf.Variable(0,dtype=tf.int32,trainable=False,name="global_step") #tf.train.get_or_create_global_step()

	optimizer = tf.train.AdamOptimizer(learning_rate)

	# Wrap the optimizer in SyncReplicasOptimizer in order
	# to have synchronous SGD. Otherwise, it will be asynchronous
	if FLAGS.is_sync:
		optimizer = tf.train.SyncReplicasOptimizer(optimizer,
							replicas_to_aggregate=num_replicas,
							total_num_replicas=num_replicas)


	model["optimizer"] = optimizer

	model["train_op"] = optimizer.minimize(model["loss"], model["global_step"])

	"""
	Summaries for TensorBoard
	"""
	tf.summary.scalar("loss", model["loss"])
	tf.summary.histogram("loss", model["loss"])
	tf.summary.scalar("dice", model["metric_dice"])
	tf.summary.histogram("dice", model["metric_dice"])

	tf.summary.scalar("sensitivity", model["metric_sensitivity"])
	tf.summary.histogram("sensitivity", model["metric_sensitivity"])
	tf.summary.scalar("specificity", model["metric_specificity"])
	tf.summary.histogram("specificity", model["metric_specificity"])

	tf.summary.image("predictions", predictionMask, max_outputs=settings_dist.TENSORBOARD_IMAGES)
	tf.summary.image("ground_truth", msks, max_outputs=settings_dist.TENSORBOARD_IMAGES)
	tf.summary.image("images", imgs, max_outputs=settings_dist.TENSORBOARD_IMAGES)

	return model
