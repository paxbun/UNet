# Copyright (c) 2021 Chanjung Kim. All rights reserved.
# Licensed under the MIT License.

from absl import app
from absl import flags
from pathlib import Path
from os import path, listdir

from unet.model import UNet
from dataset import Provider

from tqdm import tqdm

import tensorflow_io as tfio
import soundfile as sf

FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint", None,
                    "Directory where weights are saved", required=True)
flags.DEFINE_string("dataset_path", f"{Path.home()}/musdb18", "Dataset Path")
flags.DEFINE_string("stem", "vocals", "Stem to separate")
flags.DEFINE_integer("num_songs", 3, "Number of songs to use")


def main(argv):
    model = UNet.make()
    dataset = Provider(FLAGS.dataset_path,
                       FLAGS.stem,
                       subsets="test",
                       max_decoded=FLAGS.num_songs,
                       stft=False)
    checkpoint_dir = FLAGS.checkpoint

    if path.exists(checkpoint_dir):
        checkpoints = [name for name in listdir(
            checkpoint_dir) if "ckpt" in name]
        checkpoints.sort()
        checkpoint_name = checkpoints[-1].split(".")[0]
        model.load_weights(f"{checkpoint_dir}/{checkpoint_name}.ckpt")

    tracks = list(dataset.generate(FLAGS.num_songs, 1))
    sdr_sum = 0.0
    for idx, track in tqdm(enumerate(tracks)):
        result = [model.predict(stft) for stft in track.get_mixed_stft()]
        sdr, result = track.compare_predict_truth(result)
        sdr_sum += sdr
        result = tfio.audio.resample(result, 8192, 44100)
        sf.write(f"output{idx}.wav", result, 44100)

    result = sdr_sum / len(tracks)
    print("\n\n\n\n\nSDR:", result)


if __name__ == '__main__':
    app.run(main)
