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
        model = UNet()
        model.compile(optimizer=optimizer, loss=loss)
        model.build(input_shape=(None, 512, 128, 2))
        return model

    def __init__(self):
        self.encoders = []
        self.decoders = []
        for i in range(5):
            self.encoders.append(Encoder(16 * (2 ** i)))
            self.decoders.append(Decoder(16 * (2 ** i), i >= 2))
        self.last_encoder = Encoder(512)
        self.mask = tf.keras.layers.Multiply()

    def call(self, inputs):
        encoder_output = inputs
        encoder_outputs = []
        for i in range(5):
            encoder_output = self.encoders[i](encoder_output)
            encoder_outputs.append(encoder_output)
        decoder_output = self.last_encoder(encoder_output)
        for i in range(4, -1, -1):
            decoder_output = self.decoders[i]([
                encoder_output[i],
                decoder_output
            ])
        output = self.mask([inputs, decoder_output])
        return output
