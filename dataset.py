# Copyright (c) 2021 Chanjung Kim. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
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

    __slots__ = "length", "mixed", "stem"

    @staticmethod
    def from_track(track, stem: str):
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
        return DecodedTrack(mixed_li, stem_li)

    def __init__(self, mixed: list, stem: list):
        self.mixed = mixed
        self.stem = stem


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
    """

    STEMS = "vocals", "drums", "bass", "other"

    def __init__(self, root: str, stem: str, subsets: Union[str, List[str]] = "train", max_decoded: int = 100):
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
                    self.tracks[idx], self.stem)
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
                for m, s in zip(track.mixed, track.stem):
                    yield m[0], s[0]
                    yield m[1], s[1]

    def make_dataset(self, num_songs: int, repeat: int) -> tf.data.Dataset:
        output_types = (tf.float32, tf.float32)
        output_shapes = (
            tf.TensorShape((None, 128, 513, 2)),
            tf.TensorShape((None, 128, 513, 2)))
        return tf.data.Dataset.from_generator(lambda: self.generate(num_songs, repeat),
                                              output_types=output_types,
                                              output_shapes=output_shapes)
