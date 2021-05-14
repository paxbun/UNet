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


class DatasetParam:
    """Contains parameters for dataset generation.

    Attributes:
        num_songs: Total number of songs
        num_samples: Total number of samples in one batch
        repeat: Number of repeats
    """

    __slots__ = "num_songs", "num_samples", "repeat"

    def __init__(self,
                 num_songs: int = 100,
                 num_samples: int = 100,
                 repeat: int = 400):

        self.num_songs = num_songs
        self.num_samples = num_samples
        self.repeat = repeat


class DecodedTrack:
    """Contains decoded audio from the database.

    Attributes:
        length: Number of samples
        mixed: A tuple of numpy arrays from the mixture
        stems: Dictionary where the key is the name of the stem and the value is a tuple of numpy arrays from the stem
    """

    __slots__ = "length", "mixed", "stems"

    @staticmethod
    def from_track(track, stem: str):
        mixed_audio = tfio.audio.resample(track.audio, 44100, 8192)
        mixed = (mixed_audio[:, 0], mixed_audio[:, 1])

        stem_audio = tfio.audio.resample(
            track.targets[stem].audio, 44100, 8192)
        stem = (stem_audio[:, 0], stem_audio[:, 1])

        length = mixed[0].shape[-1]
        return DecodedTrack(length, mixed, stem)

    def __init__(self,
                 length: int,
                 mixed: Tuple[np.ndarray, np.ndarray],
                 stem: Tuple[np.ndarray, np.ndarray]):
        self.length = length
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

    def generate(self, p: DatasetParam):
        indices = list(range(self.num_tracks))
        random.shuffle(indices)
        indices = indices[:p.num_songs]
        self.decode(indices)

        duration = 1024 + 768 * 127

        # Make `p.repeat` batches
        for _ in range(p.repeat):
            x_batch = np.zeros((p.num_samples * 2, duration))
            y_batch = np.zeros((p.num_samples * 2, duration))

            for i in range(p.num_samples):
                track = self.decoded[random.choice(indices)]
                begin = random.randint(0, track.length - duration)
                end = begin + duration
                left = i * 2
                right = left + 1

                x_batch[left] = track.mixed[0][begin:end]
                x_batch[right] = track.mixed[1][begin:end]
                y_batch[left] = track.stem[0][begin:end]
                y_batch[right] = track.stem[1][begin:end]

            x_batch = tf.signal.stft(x_batch, 1024, 768)
            y_batch = tf.signal.stft(y_batch, 1024, 768)

            x_batch = tf.concat([tf.real(x_batch), tf.imag(x_batch)], axis=-1)
            y_batch = tf.concat([tf.real(y_batch), tf.imag(y_batch)], axis=-1)

            yield x_batch, y_batch

    def make_dataset(self, p: DatasetParam) -> tf.data.Dataset:
        output_types = (tf.float32, tf.float32)
        output_shapes = (
            tf.TensorShape(
                (p.num_samples * 2, 512, 128, 2)),
            tf.TensorShape(
                (p.num_samples * 2, 512, 128, 2)))
        return tf.data.Dataset.from_generator(lambda: self.generate(p),
                                              output_types=output_types,
                                              output_shapes=output_shapes)
