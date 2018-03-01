import os.path
import tensorflow as tf

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



CHANNEL_LAST = True
if CHANNEL_LAST:
	concat_axis = -1
	data_format = 'channels_last'

else:
	concat_axis = 1
	data_format = 'channels_first'

tf.keras.backend.set_image_data_format(data_format)

def define_model(input_tensor, use_upsampling=False, n_cl_out=1, dropout=0.2, print_summary = False):


	# Set keras learning phase to train
	tf.keras.backend.set_learning_phase(True)

	# Don't initialize variables on the fly
	tf.keras.backend.manual_variable_initialization(False)

	inputs = tf.keras.layers.Input(tensor=input_tensor, name='Images')

	params = dict(kernel_size=(3, 3, 3), activation='relu',
				  padding='same', data_format=data_format,
				  kernel_initializer='he_uniform') #RandomUniform(minval=-0.01, maxval=0.01, seed=816))

	conv1 = tf.keras.layers.Conv3D(name='conv1a', filters=32, **params)(inputs)
	conv1 = tf.keras.layers.Conv3D(name='conv1b', filters=64, **params)(conv1)
	pool1 = tf.keras.layers.MaxPooling3D(name='pool1', pool_size=(2, 2, 2))(conv1)

	conv2 = tf.keras.layers.Conv3D(name='conv2a', filters=64, **params)(pool1)
	conv2 = tf.keras.layers.Conv3D(name='conv2b', filters=128, **params)(conv2)
	pool2 = tf.keras.layers.MaxPooling3D(name='pool2', pool_size=(2, 2, 2))(conv2)

	conv3 = tf.keras.layers.Conv3D(name='conv3a', filters=128, **params)(pool2)
	conv3 = tf.keras.layers.Dropout(dropout)(conv3) ### Trying dropout layers earlier on, as indicated in the paper
	conv3 = tf.keras.layers.Conv3D(name='conv3b', filters=256, **params)(conv3)
	pool3 = tf.keras.layers.MaxPooling3D(name='pool3', pool_size=(2, 2, 2))(conv3)

	conv4 = tf.keras.layers.Conv3D(name='conv4a', filters=256, **params)(pool3)
	conv4 = tf.keras.layers.Dropout(dropout)(conv4) ### Trying dropout layers earlier on, as indicated in the paper
	conv4 = tf.keras.layers.Conv3D(name='conv4b', filters=512, **params)(conv4)

	up4 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling3D(name='up4', size=(2, 2, 2))(conv4), conv3], axis=concat_axis)

	conv5 = tf.keras.layers.Conv3D(name='conv5a', filters=256, **params)(up4)
	conv5 = tf.keras.layers.Conv3D(name='conv5b', filters=256, **params)(conv5)

	up5 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling3D(name='up5', size=(2, 2, 2))(conv5), conv2], axis=concat_axis)

	conv6 = tf.keras.layers.Conv3D(name='conv6a', filters=128, **params)(up5)
	conv6 = tf.keras.layers.Conv3D(name='conv6b', filters=128, **params)(conv6)

	up6 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling3D(name='up6', size=(2, 2, 2))(conv6), conv1], axis=concat_axis)

	conv7 = tf.keras.layers.Conv3D(name='conv7a', filters=128, **params)(up6)
	conv7 = tf.keras.layers.Conv3D(name='conv7b', filters=128, **params)(conv7)

	mask = tf.keras.layers.Conv3D(name='Mask', filters=n_cl_out, kernel_size=(1, 1, 1),
					data_format=data_format, activation='sigmoid')(conv7)

	model = tf.keras.models.Model(inputs=[inputs], outputs=[mask])

	# optimizer=tf.keras.optimizers.Adam()
	# model.compile(optimizer=optimizer, loss=dice_coef_loss, metrics=[dice_coef])

	if print_summary:
		print (model.summary())

	return model


def sensitivity(y_true, y_pred, smooth = 1. ):

	intersection = tf.reduce_sum(y_true * y_pred)
	coef = (intersection + smooth) / (tf.reduce_sum(y_true) + smooth)
	return coef

def specificity(y_true, y_pred, smooth = 1. ):

	intersection = tf.reduce_sum(y_true * y_pred)
	coef = (intersection + smooth) / (tf.reduce_sum(y_pred) + smooth)
	return coef
