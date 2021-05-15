# Copyright (c) 2021 Chanjung Kim. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
import math
import musdb
import random
import gc
from tqdm import tqdm
from typing import Union, List, Tuple, Optional


class DecodedTrack:
    """Contains decoded audio from the database.

    Attributes:
        length: Number of samples
        mixed: A tuple of numpy arrays from the mixture
        stems: Dictionary where the key is the name of the stem and the value is a tuple of numpy arrays from the stem
    """

    __slots__ = "length", "mixed", "stem", "stft"

    @staticmethod
    def from_track(track, stem: str, stft: bool):
        if not stft:
            mixed_audio = tfio.audio.resample(
                track.audio.astype(np.float32), 44100, 8192)
            stem_audio = tfio.audio.resample(
                track.targets[stem].audio.astype(np.float32), 44100, 8192)

            length = mixed_audio.shape[0]
            new_length = (length - 256) // (768 * 128) * (768 * 128) + 256

            mixed = [mixed_audio[:new_length, 0], mixed_audio[:new_length, 1]]
            stem = [stem_audio[:new_length, 0], stem_audio[:new_length, 1]]

            return DecodedTrack(mixed, stem, stft)
        else:
            mixed_audio = tfio.audio.resample(
                track.audio.astype(np.float32), 44100, 8192)
            mixed = tf.signal.stft(
                [mixed_audio[:, 0], mixed_audio[:, 1]], 1024, 768).numpy()
            mixed = np.stack([mixed.real, mixed.imag], axis=3)

            stem_audio = tfio.audio.resample(
                track.targets[stem].audio.astype(np.float32), 44100, 8192)
            stem = tf.signal.stft(
                [stem_audio[:, 0], stem_audio[:, 1]], 1024, 768).numpy()
            stem = np.stack([stem.real, stem.imag], axis=3)

            length = mixed.shape[1]
            max_offset = length % 128
            num_samples = length // 128
            mixed_li = []
            stem_li = []

            for offset in range(0, max_offset + 1, 8):
                mixed_li.append(
                    np.reshape(mixed[:, offset:num_samples * 128 + offset], (2, num_samples, 128, 513, 2)))
                stem_li.append(
                    np.reshape(stem[:, offset:num_samples * 128 + offset], (2, num_samples, 128, 513, 2)))

            return DecodedTrack(mixed_li, stem_li, stft)

    def __init__(self, mixed, stem, stft: bool):
        self.mixed = mixed
        self.stem = stem
        self.stft = stft

    def get_mixed_stft(self):
        if self.stft:
            raise ValueError("This track is for training")

        rtn = []
        for i in range(2):
            mixed = self.mixed[i]

            mixed = tf.signal.stft(mixed, 1024, 768).numpy()
            mixed = np.stack([mixed.real, mixed.imag], axis=2)

            assert mixed.shape[0] % 128 == 0
            num_samples = mixed.shape[0] // 128

            rtn.append(np.reshape(mixed, (num_samples, 128, 513, 2)))

        return rtn

    def compare_predict_truth(self, pred):
        """Returns SDR.
        """

        if self.stft:
            raise ValueError("This track is for training")

        num_samples = pred[0].shape[0]
        pred = np.reshape(pred, (2, num_samples * 128, 513, 2))
        stem = tf.signal.inverse_stft(
            tf.complex(pred[:, :, :, 0], pred[:, :, :, 1]),
            1024, 768, window_fn=tf.signal.inverse_stft_window_fn(768))
        sdr = 20 * tf.math.log(
            tf.norm(stem - self.stem) / tf.norm(self.stem)
        ) / math.log(10)
        stem = np.transpose(stem)

        return sdr, stem


class Provider:
    """Decodes audio from the database.

    Attributes:
        tracks: List of tracks
        stem: Stem to decode
        subsets: Subsets to decode
        num_tracks: Number of tracks
        decoded: List of decoded tracks
        num_decoded: Number of decoded tracks in `decoded`
        max_decoded: Maximum number of decoded tracks
        ord_decoded: The order in which each track is decoded
        next_ord: The order which will be granted to the next decoded track
        stft: STFT is applied to the resulting tracks
    """

    STEMS = "vocals", "drums", "bass", "other"

    def __init__(self, root: str, stem: str, subsets: Union[str, List[str]] = "train", max_decoded: int = 100, stft: bool = False):
        if max_decoded < 1:
            raise ValueError("max_decoded must be greater than 0")

        if stem not in Provider.STEMS:
            raise ValueError(f"'{stem}' is not a valid stem name")

        self.tracks = list(musdb.DB(root=root, subsets=subsets))
        self.stem = stem
        self.subsets = subsets
        self.num_tracks = len(self.tracks)
        self.decoded: List[Optional[DecodedTrack]] = [None] * self.num_tracks
        self.num_decoded = 0
        self.max_decoded = max_decoded
        self.ord_decoded = [-1] * self.num_tracks
        self.next_ord = 0
        self.stft = stft

    def remove_oldest(self):
        assert self.num_decoded > 0
        idx = None
        for i in range(self.num_tracks):
            if self.ord_decoded[i] != -1:
                if idx == None or self.ord_decoded[idx] > self.ord_decoded[i]:
                    idx = i
        assert idx != None
        self.decoded[idx] = None
        self.num_decoded -= 1
        self.ord_decoded[idx] = -1
        gc.collect()

    def decode(self, indices: Union[int, List[int]]):
        if type(indices) == int:
            indices = [indices]
        if len(indices) > self.max_decoded:
            raise ValueError("Cannot decode more than `max_decoded` tracks")

        for idx in indices:
            if self.decoded[idx] != None:
                self.ord_decoded[idx] = self.next_ord
                self.next_ord += 1

        indices = [idx for idx in indices if self.decoded[idx] == None]
        if indices:
            print(f"\n\nDecoding Audio {indices} of subset {self.subsets}...")
            for idx in tqdm(indices):
                while self.num_decoded >= self.max_decoded:
                    self.remove_oldest()
                self.decoded[idx] = DecodedTrack.from_track(
                    self.tracks[idx], self.stem, self.stft)
                self.num_decoded += 1
                self.ord_decoded[idx] = self.next_ord
                self.next_ord += 1

    def generate(self, num_songs: int, repeat: int):
        indices = list(range(self.num_tracks))
        random.shuffle(indices)
        indices = indices[:num_songs]
        self.decode(indices)

        for _ in range(repeat):
            for index in indices:
                track = self.decoded[index]
                if track.stft:
                    for m, s in zip(track.mixed, track.stem):
                        yield m[0], s[0]
                        yield m[1], s[1]
                else:
                    yield track

    def make_dataset(self, num_songs: int, repeat: int) -> tf.data.Dataset:
        if not self.stft:
            raise ValueError(
                "make_dataset can be called iff self.stft is True")

        output_types = (tf.float32, tf.float32)
        output_shapes = (
            tf.TensorShape((None, 128, 513, 2)),
            tf.TensorShape((None, 128, 513, 2)))
        return tf.data.Dataset.from_generator(lambda: self.generate(num_songs, repeat),
                                              output_types=output_types,
                                              output_shapes=output_shapes)
