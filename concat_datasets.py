import tensorflow as tf

from tensorflow.keras import mixed_precision

import numpy as np

from tensorflow import keras
from tensorflow.data import Dataset
from tensorflow.keras import layers
import tensorflow_datasets as tfds

import matplotlib.pyplot as plt

from common_functions import *
from tqdm import tqdm

setup_CUDA(False, "-1")

def get_element_shape(dataset):
    for element in dataset:
        return element['strain'].shape[1:]

if __name__ == "__main__":

    # User parameters:
    noise_paths  = ["noise_0", "noise_1", "noise_2", "noise_3",  "noise_4", "noise_5",  "noise_6", "noise_7", "noise_8",  "noise_9"]
    signal_paths = ["cbc_10_0", "cbc_10_1", "cbc_10_2","cbc_10_3", "cbc_10_4", "cbc_10_5", "cbc_10_6", "cbc_10_7", "cbc_10_8", "cbc_10_9"]
    
    split_datasets = {}
    
    data = []
    labels = []
    for path in tqdm(noise_paths):
        dataset = tf.data.experimental.load(f"../skywarp_data/datasets/{path}")

        for i in dataset.as_numpy_iterator():
            data.append(i)
            labels.append(0)
    
    for path in tqdm(signal_paths):
        dataset = tf.data.experimental.load(f"../skywarp_data/datasets/{path}")

        for i in dataset.as_numpy_iterator():
            data.append(i)
            labels.append(1)

    dataset = tf.data.Dataset.from_tensor_slices(
        {"strain": data, 
         "signal_present": labels})

    split_datasets["train"] = dataset
    
    # Optionally define a custom `data_dir`.
    # If None, then the default data dir is used.
    custom_data_dir = "../skywarp_data/skywarp_dataset_gaussian"

    # Define the builder.
    builder = tfds.dataset_builders.TfDataBuilder(
        name="skywarp_dataset_gaussian",
        config="strain_and_label",
        version="0.0.1",
        data_dir=custom_data_dir,
        split_datasets = split_datasets,
        features=tfds.features.FeaturesDict({
            "strain": tfds.features.Tensor(shape = (len(data[0]),), dtype=tf.float16),
            "signal_present": tfds.features.ClassLabel(num_classes = 2),
        }),
        description="Dataset of CBC signals and noise.",
        release_notes={
            "0.0.1": "Test Dataset",
        }
    )

    # Make the builder store the data as a TFDS dataset.
    builder.download_and_prepare()