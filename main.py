# Copyright (c) 2021 Chanjung Kim. All rights reserved.
# Licensed under the MIT License.

from absl import app
from absl import flags
from pathlib import Path
from os import path, listdir

from unet.model import UNet
from dataset import Provider

FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint", None,
                    "Directory to save weights", required=True)
flags.DEFINE_string("dataset_path", f"{Path.home()}/musdb18", "Dataset Path")
flags.DEFINE_string("stem", "vocals", "Stem to separate")
flags.DEFINE_integer("epochs", None, "Number of epochs to repeat")
flags.DEFINE_integer(
    "num_songs", 20, "Number of songs to get samples from for each epoch")
flags.DEFINE_integer(
    "val_num_songs", 3, "Number of songs for validation from for each epoch")
flags.DEFINE_integer(
    "max_decoded", 100, "Number of decoded songs present in the memory")
flags.DEFINE_integer(
    "val_max_decoded", 100, "Number of decoded songs for validation present in the memory")


def main(argv):
    model = UNet.make()
    dataset = Provider(FLAGS.dataset_path, FLAGS.stem,
                       max_decoded=FLAGS.max_decoded)
    val_dataset = Provider(
        FLAGS.dataset_path, FLAGS.stem, subsets="test", max_decoded=FLAGS.val_max_decoded)
    checkpoint_dir = FLAGS.checkpoint

    epoch = 0
    if path.exists(checkpoint_dir):
        checkpoints = [name for name in listdir(
            checkpoint_dir) if "ckpt" in name]
        checkpoints.sort()
        checkpoint_name = checkpoints[-1].split(".")[0]
        epoch = int(checkpoint_name) + 1
        model.load_weights(f"{checkpoint_dir}/{checkpoint_name}.ckpt")

    epochs_to_inc = FLAGS.epochs
    while epochs_to_inc == None or epochs_to_inc > 0:
        print(f"Epoch: {epoch}")
        train = dataset.make_dataset(FLAGS.num_songs, 20)
        val = val_dataset.make_dataset(FLAGS.val_num_songs, 1)
        history = model.fit(train, validation_data=val)
        model.save_weights(f"{checkpoint_dir}/{epoch:05d}.ckpt")
        epoch += 1
        if epochs_to_inc != None:
            epochs_to_inc -= 1
        model.save(f"{checkpoint_dir}/model")


if __name__ == '__main__':
    app.run(main)
