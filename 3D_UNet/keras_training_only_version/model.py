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

import keras as K


def dice_coef(y_true, y_pred, axis=(1, 2, 3), smooth=1.):
    """
    Sorenson (Soft) Dice
    2 * |TP| / |T|*|P|
    where T is ground truth mask and P is the prediction mask
    """
    intersection = tf.reduce_sum(y_true * y_pred, axis=axis)
    union = tf.reduce_sum(y_true + y_pred, axis=axis)
    numerator = tf.constant(2.) * intersection + smooth
    denominator = union + smooth
    coef = numerator / denominator

    return tf.reduce_mean(coef)


def dice_coef_loss(target, prediction, axis=(1, 2, 3), smooth=1.):
    """
    Sorenson (Soft) Dice loss
    Using -log(Dice) as the loss since it is better behaved.
    Also, the log allows avoidance of the division which
    can help prevent underflow when the numbers are very small.
    """
    intersection = tf.reduce_sum(prediction * target, axis=axis)
    p = tf.reduce_sum(prediction, axis=axis)
    t = tf.reduce_sum(target, axis=axis)
    numerator = tf.reduce_mean(intersection + smooth)
    denominator = tf.reduce_mean(t + p + smooth)
    dice_loss = -tf.log(2.*numerator) + tf.log(denominator)

    return dice_loss


def combined_dice_ce_loss(target, prediction, axis=(1, 2, 3), smooth=1., weight=.7):
    """
    Combined Dice and Binary Cross Entropy Loss
    """
    return weight*dice_coef_loss(target, prediction, axis, smooth) + \
        (1-weight)*K.losses.binary_crossentropy(target, prediction)


CHANNEL_LAST = True
if CHANNEL_LAST:
    concat_axis = -1
    data_format = "channels_last"

else:
    concat_axis = 1
    data_format = "channels_first"


def unet_3d(input_shape, use_upsampling=False, learning_rate=0.001,
            n_cl_in=1, n_cl_out=1, dropout=0.2, print_summary=False):
    """
    3D U-Net
    """

    inputs = K.layers.Input(shape=input_shape, name="MRImages")

    params = dict(kernel_size=(3, 3, 3), activation=None,
                  padding="same", data_format=data_format,
                  kernel_initializer="he_uniform")

    # Transposed convolution parameters
    params_trans = dict(data_format=data_format,
                        kernel_size=(2, 2, 2), strides=(2, 2, 2),
                        padding="same")

    fms = 16  #32 or 16 depending on your memory size

    encodeA = K.layers.Conv3D(name="encodeAa", filters=fms, **params)(inputs)
    encodeA = K.layers.BatchNormalization()(encodeA)
    encodeA = K.layers.Activation("relu")(encodeA)
    encodeA = K.layers.Conv3D(name="encodeAb", filters=fms, **params)(encodeA)
    encodeA = K.layers.BatchNormalization()(encodeA)
    encodeA = K.layers.Activation("relu")(encodeA)
    poolA = K.layers.MaxPooling3D(name="poolA", pool_size=(2, 2, 2))(encodeA)

    encodeB = K.layers.Conv3D(name="encodeBa", filters=fms*2, **params)(poolA)
    encodeB = K.layers.BatchNormalization()(encodeB)
    encodeB = K.layers.Activation("relu")(encodeB)
    encodeB = K.layers.Conv3D(name="encodeBb", filters=fms*2, **params)(encodeB)
    poolB = K.layers.MaxPooling3D(name="poolB", pool_size=(2, 2, 2))(encodeB)

    encodeC = K.layers.Conv3D(name="encodeCa", filters=fms*4, **params)(poolB)
    encodeC = K.layers.BatchNormalization()(encodeC)
    encodeC = K.layers.Activation("relu")(encodeC)
    encodeC = K.layers.SpatialDropout3D(dropout,
                                        data_format=data_format)(encodeC)
    encodeC = K.layers.Conv3D(name="encodeCb", filters=fms*4, **params)(encodeC)
    encodeC = K.layers.BatchNormalization()(encodeC)
    encodeC = K.layers.Activation("relu")(encodeC)

    poolC = K.layers.MaxPooling3D(name="poolC", pool_size=(2, 2, 2))(encodeC)

    encodeD = K.layers.Conv3D(name="encodeDa", filters=fms*8, **params)(poolC)
    encodeD = K.layers.BatchNormalization()(encodeD)
    encodeD = K.layers.Activation("relu")(encodeD)
    encodeD = K.layers.SpatialDropout3D(dropout,
                                        data_format=data_format)(encodeD)
    encodeD = K.layers.Conv3D(name="encodeDb", filters=fms*8, **params)(encodeD)
    encodeD = K.layers.BatchNormalization()(encodeD)
    encodeD = K.layers.Activation("relu")(encodeD)

    poolD = K.layers.MaxPooling3D(name="poolD", pool_size=(2, 2, 2))(encodeD)

    encodeE = K.layers.Conv3D(name="encodeEa", filters=fms*16, **params)(poolD)
    encodeE = K.layers.BatchNormalization()(encodeE)
    encodeE = K.layers.Activation("relu")(encodeE)
    encodeE = K.layers.Conv3D(name="encodeEb", filters=fms*16, **params)(encodeE)
    encodeE = K.layers.BatchNormalization()(encodeE)
    encodeE = K.layers.Activation("relu")(encodeE)

    if use_upsampling:
        up = K.layers.UpSampling3D(name="upE", size=(2, 2, 2),
                                   interpolation="bilinear")(encodeE)
    else:
        up = K.layers.Conv3DTranspose(name="transconvE", filters=fms*8,
                                      **params_trans)(encodeE)
    concatD = K.layers.concatenate([up, encodeD], axis=concat_axis, name="concatD")

    decodeC = K.layers.Conv3D(name="decodeCa", filters=fms*8, **params)(concatD)
    decodeC = K.layers.BatchNormalization()(decodeC)
    decodeC = K.layers.Activation("relu")(decodeC)
    decodeC = K.layers.Conv3D(name="decodeCb", filters=fms*8, **params)(decodeC)
    decodeC = K.layers.BatchNormalization()(decodeC)
    decodeC = K.layers.Activation("relu")(decodeC)

    if use_upsampling:
        up = K.layers.UpSampling3D(name="upC", size=(2, 2, 2),
                                   interpolation="bilinear")(decodeC)
    else:
        up = K.layers.Conv3DTranspose(name="transconvC", filters=fms*4,
                                      **params_trans)(decodeC)
    concatC = K.layers.concatenate([up, encodeC], axis=concat_axis, name="concatC")

    decodeB = K.layers.Conv3D(name="decodeBa", filters=fms*4, **params)(concatC)
    decodeB = K.layers.BatchNormalization()(decodeB)
    decodeB = K.layers.Activation("relu")(decodeB)
    decodeB = K.layers.Conv3D(name="decodeBb", filters=fms*4, **params)(decodeB)
    decodeB = K.layers.BatchNormalization()(decodeB)
    decodeB = K.layers.Activation("relu")(decodeB)

    if use_upsampling:
        up = K.layers.UpSampling3D(name="upB", size=(2, 2, 2),
                                   interpolation="bilinear")(decodeB)
    else:
        up = K.layers.Conv3DTranspose(name="transconvB", filters=fms*2,
                                      **params_trans)(decodeB)
    concatB = K.layers.concatenate([up, encodeB], axis=concat_axis, name="concatB")

    decodeA = K.layers.Conv3D(name="decodeAa", filters=fms*2, **params)(concatB)
    decodeA = K.layers.BatchNormalization()(decodeA)
    decodeA = K.layers.Activation("relu")(decodeA)
    decodeA = K.layers.Conv3D(name="decodeAb", filters=fms*2, **params)(decodeA)
    decodeA = K.layers.BatchNormalization()(decodeA)
    decodeA = K.layers.Activation("relu")(decodeA)

    if use_upsampling:
        up = K.layers.UpSampling3D(name="upA", size=(2, 2, 2),
                                   interpolation="bilinear")(decodeA)
    else:
        up = K.layers.Conv3DTranspose(name="transconvA", filters=fms,
                                      **params_trans)(decodeA)
    concatA = K.layers.concatenate([up, encodeA], axis=concat_axis, name="concatA")

    convOut = K.layers.Conv3D(name="convOuta", filters=fms, **params)(concatA)
    convOut = K.layers.BatchNormalization()(convOut)
    convOut = K.layers.Activation("relu")(convOut)
    convOut = K.layers.Conv3D(name="convOutb", filters=fms, **params)(convOut)
    convOut = K.layers.BatchNormalization()(convOut)
    convOut = K.layers.Activation("relu")(convOut)

    prediction = K.layers.Conv3D(name="PredictionMask",
                                 filters=n_cl_out, kernel_size=(1, 1, 1),
                                 data_format=data_format,
                                 activation="sigmoid")(convOut)

    model = K.models.Model(inputs=[inputs], outputs=[prediction])

    if print_summary:
        model.summary()

    opt = K.optimizers.Adam(learning_rate)

    return model, opt


def sensitivity(target, prediction, axis=(1, 2, 3), smooth=1.):
    """
    Sensitivity
    """
    intersection = tf.reduce_sum(prediction * target, axis=axis)
    coef = (intersection + smooth) / (tf.reduce_sum(target,
                                                    axis=axis) + smooth)
    return tf.reduce_mean(coef)


def specificity(target, prediction, axis=(1, 2, 3), smooth=1.):
    """
    Specificity
    """
    intersection = tf.reduce_sum(prediction * target, axis=axis)
    coef = (intersection + smooth) / (tf.reduce_sum(prediction,
                                                    axis=axis) + smooth)
    return tf.reduce_mean(coef)
