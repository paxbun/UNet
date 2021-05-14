# Copyright (c) 2021 Chanjung Kim. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf


class Encoder(tf.keras.layers.Layer):
    """Represents an encoder block (p. 746).

    Attributes:
        num_filters: Number of filters (see Figure 1 in p. 747)
    """

    def __init__(self, num_filters: int):
        super(Encoder, self).__init__()
        self.num_filters = num_filters
        self.padding = tf.keras.layers.ZeroPadding2D((2, 2))
        self.conv = tf.keras.layers.Conv2D(
            num_filters, kernel_size=(5, 5), strides=(2, 2))
        self.norm = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.LeakyReLU(0.2)

    def call(self, inputs):
        output = self.padding(inputs)
        output = self.conv(output)
        output = self.norm(output)
        output = self.relu(output)
        return output

    def compute_output_signature(self, input_signature: tf.TensorSpec) -> tf.TensorSpec:
        output = self.padding.compute_output_signature(input_signature)
        output = self.conv.compute_output_signature(output)
        return output

    def compute_output_shape(self, input_shape):
        output = self.padding.compute_output_shape(input_shape)
        output = self.conv.compute_output_shape(output)
        return output

    def get_config(self):
        return {"num_filters": self.num_filters}


class Decoder(tf.keras.layers.Layer):
    """Represents a decoder block (p. 746).

    Attributes:
        num_filters: Number of filters (see Figure 1 in p. 747)
        has_dropout: Indicates that this block has a dropout layer (p. 746)
    """

    def __init__(self, num_filters: int, has_dropout: bool):
        super(Decoder, self).__init__()
        self.num_filters = num_filters
        self.dconv = tf.keras.layers.Conv2DTranspose(
            num_filters, kernel_size=(5, 5), strides=(2, 2))
        self.crop = tf.keras.layers.Cropping2D(((2, 1), (2, 1)))
        self.norm = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.has_dropout = has_dropout
        if has_dropout:
            self.dropout = tf.keras.layers.Dropout(0.5)

    def call(self, inputs):
        output = self.dconv(inputs)
        output = self.crop(output)
        output = self.norm(output)
        output = self.relu(output)
        if self.has_dropout:
            output = self.dropout(output)
        return output

    def compute_output_signature(self, input_signature: tf.TensorSpec) -> tf.TensorSpec:
        output = self.dconv.compute_output_signature(input_signature)
        output = self.crop.compute_output_signature(output)
        return output

    def compute_output_shape(self, input_shape):
        output = self.dconv.compute_output_shape(input_shape)
        output = self.crop.compute_output_shape(output)
        return output

    def get_config(self):
        return {
            "num_filters": self.num_filters,
            "has_dropout": self.has_dropout
        }
