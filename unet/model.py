# Copyright (c) 2021 Chanjung Kim. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf

from .layers import Encoder, Decoder


class UNet(tf.keras.Model):
    """Implements U-Net.

    """

    @staticmethod
    def make(optimizer: tf.keras.optimizers.Optimizer = "adam",
             loss: tf.keras.losses.Loss = tf.keras.losses.MeanAbsoluteError()):

        input = tf.keras.Input(shape=(512, 128, 2), dtype=tf.float32)
        encoder_output = input
        encoder_outputs = []
        for i in range(5):
            encoder_output = Encoder(16 * (2 ** i))(encoder_output)
            encoder_outputs.append(encoder_output)
        decoder_output = Encoder(512)(encoder_output)
        for i in range(4, -1, -1):
            decoder_output = Decoder(16 * (2 ** i), i >= 2)(decoder_output)
            decoder_output = tf.concat(
                [decoder_output, encoder_outputs[i]], axis=-1)
        decoder_output = tf.keras.layers.Conv2DTranspose(
            2, kernel_size=(5, 5), strides=(2, 2), activation="sigmoid")(decoder_output)
        decoder_output = tf.keras.layers.Cropping2D(
            ((2, 1), (2, 1)))(decoder_output)
        output = tf.keras.layers.Multiply()([input, decoder_output])
        model = tf.keras.Model(inputs=[input], outputs=[output])
        model.compile(optimizer=optimizer, loss=loss)

        return model
