import tensorflow as tf

import numpy as np
import logging

from tensorflow import keras
from tensorflow.data import Dataset
from tensorflow.keras import layers

from py_ml_tools.dataset import get_ifo_data, O3, get_ifo_data_generator
from py_ml_tools.setup import setup_cuda

from tensorflow.keras import mixed_precision
import matplotlib.pyplot as plt

import h5py

from tensorflow.keras import backend as K

from tensorflow.data.experimental import AutoShardPolicy
import gc

from tqdm import tqdm
from common_functions import *

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

def save_data_to_hdf5(model_name, noise_type, roc_data, efficiency_scores, far_scores):
    with h5py.File(f'../skywarp_data/{model_name}_{noise_type}_data.h5', 'w') as h5f:
        # Save ROC data
        roc_group = h5f.create_group('roc_data')
        roc_group.create_dataset('fpr', data=roc_data['fpr'])
        roc_group.create_dataset('tpr', data=roc_data['tpr'])
        roc_group.attrs['roc_auc'] = roc_data['roc_auc']

        # Save efficiency scores
        eff_group = h5f.create_group('efficiency_scores')
        for i, score in enumerate(efficiency_scores):
            eff_group.create_dataset(f'score_{i}', data=score)

        # Save FAR scores
        far_group = h5f.create_group('far_scores')
        far_group.create_dataset('scores', data=far_scores)

@tf.function
def roc_curve_and_auc(y_true, y_scores, chunk_size=500):
    num_thresholds = 1000
     # Use logspace with a range between 0 and 6, which corresponds to values between 1 and 1e-6
    log_thresholds = tf.exp(tf.linspace(0, -6, num_thresholds))
    # Generate thresholds focusing on values close to 1
    thresholds = 1 - log_thresholds
    
    thresholds = tf.cast(thresholds, tf.float32)
    y_true = tf.cast(y_true, tf.float32)

    num_samples = y_true.shape[0]
    num_chunks = (num_samples + chunk_size - 1) // chunk_size

    # Initialize accumulators for true positives, false positives, true negatives, and false negatives
    tp_acc = tf.zeros(num_thresholds, dtype=tf.float32)
    fp_acc = tf.zeros(num_thresholds, dtype=tf.float32)
    fn_acc = tf.zeros(num_thresholds, dtype=tf.float32)
    tn_acc = tf.zeros(num_thresholds, dtype=tf.float32)

    # Process data in chunks
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, num_samples)

        y_true_chunk = y_true[start_idx:end_idx]
        y_scores_chunk = y_scores[start_idx:end_idx]

        y_pred = tf.expand_dims(y_scores_chunk, 1) >= thresholds
        y_pred = tf.cast(y_pred, tf.float32)

        y_true_chunk = tf.expand_dims(y_true_chunk, axis=-1)
        tp = tf.reduce_sum(y_true_chunk * y_pred, axis=0)
        fp = tf.reduce_sum((1 - y_true_chunk) * y_pred, axis=0)
        fn = tf.reduce_sum(y_true_chunk * (1 - y_pred), axis=0)
        tn = tf.reduce_sum((1 - y_true_chunk) * (1 - y_pred), axis=0)

        # Update accumulators
        tp_acc += tp
        fp_acc += fp
        fn_acc += fn
        tn_acc += tn

    tpr = tp_acc / (tp_acc + fn_acc)
    fpr = fp_acc / (fp_acc + tn_acc)

    auc = tf.reduce_sum((fpr[:-1] - fpr[1:]) * (tpr[:-1] + tpr[1:])) / 2

    return fpr, tpr, auc

def concat_labels(x, y):
    return tf.concat([x, y], axis=0)

def calculate_roc_data(model, dataset):
    # Use .map() to extract the true labels and model inputs
    x_dataset = dataset.map(lambda x, y: x)
    y_true_dataset = dataset.map(lambda x, y: y)

    # Convert the true labels dataset to a tensor using reduce
    y_true = y_true_dataset.reduce(tf.constant([], dtype=tf.float32), concat_labels)

    # Get the model predictions
    y_scores = model.predict(x_dataset, verbose = 1)[:, 1]

    # Calculate the ROC curve and AUC
    fpr, tpr, roc_auc = roc_curve_and_auc(y_true, y_scores)

    return fpr.numpy(), tpr.numpy(), roc_auc.numpy()

def calculate_efficiency_scores(
        model, 
        path_suffix, 
        batch_size, 
        options
    ):
    
    snr_levels = np.linspace(0.0, 10.0, 21)
    snr_datasets = []
    
    for snr_level in snr_levels:
        path = [f"{path_suffix}_{snr_level}_e"]
        dataset = load_datasets(path).with_options(options)
        snr_datasets.append(dataset.batch(batch_size))

    # Concatenate all datasets into a single large dataset
    combined_dataset = snr_datasets[0].concatenate(snr_datasets[1])
    for ds in snr_datasets[2:]:
        combined_dataset = combined_dataset.concatenate(ds)

    # Process all examples in one go
    combined_scores = model.predict(combined_dataset, verbose = 1)

    # Split predictions back into separate arrays for each SNR level
    scores = []
    start_index = 0
    for ds in snr_datasets:
        ds_size = tf.data.experimental.cardinality(ds).numpy() * batch_size  # Calculate the total number of examples in the dataset
        scores.append(combined_scores[start_index:start_index + ds_size])
        start_index += ds_size

    return scores

def calculate_far_scores(model, noise_ds, num_examples=1E5):
    num_steps = int(num_examples // batch_size)
    
    # Predict the scores and get the second column ([:, 1])
    far_scores = model.predict(noise_ds, steps = num_steps, verbose=1)[:, 1]
    
    return far_scores

def load_cbc_datasets(data_directory_path, num_to_load):
    dataset_prefix = f"{data_directory_path}/datasets/cbc"
    dataset_paths = [f"{dataset_prefix}_{i}_v" for i in range(num_to_load)]
    
    # Check for existing dataset paths
    existing_paths = []
    for path in dataset_paths:
        if os.path.exists(path):
            existing_paths.append(path)
        else:
            print(f"Warning: {path} does not exist.")
    
    # Load the datasets with existing paths
    dataset = load_label_datasets(existing_paths, 1)
    
    return dataset

def get_element_shape(dataset):
    for element in dataset.take(1):
        return element[0].shape

def gaussian_noise_generator(num_samples=8192):
    while True:
        noise = tf.random.normal([num_samples], dtype=tf.float16)
        constant = tf.constant(0.0, dtype=tf.float32)
        yield noise, constant
        
def add_noise_label(element):
    return element, tf.constant(0.0, shape = element.shape, dtype=tf.float32)
                            
if __name__ == "__main__":
    # User parameters:
    skywarp_data_directory = "../skywarp_data/"
    
    batch_size    = 32
    num_far_tests = int(1E4)
    sample_rate_hertz = 8192.0
    example_duration_seconds = 1.0
    
    model_names = [
        "skywarp_attention_regular", 
        "skywarp_conv_attention_regular", 
        "skywarp_conv_attention_single", 
        "skywarp_conv_regular"
    ]
        
    # Load datasets:
    strategy = setup_cuda(True, "4,5,6,7")
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = AutoShardPolicy.DATA
    
    logging.basicConfig(level=logging.INFO)
        
    with strategy.scope():
        cbc_ds = load_cbc_datasets(skywarp_data_directory, 1)
        num_samples = get_element_shape(cbc_ds)[0]
        
         # Create TensorFlow dataset from the generator
        noise_ds = tf.data.Dataset.from_generator(
            generator=lambda: gaussian_noise_generator(num_samples=num_samples),
            output_signature=(
                tf.TensorSpec(shape=(num_samples,), dtype=tf.float16),
                tf.TensorSpec(shape=(), dtype=tf.float32),
            )
        ).batch(batch_size)
        
        # Create TensorFlow dataset from the generator
        real_noise_ds = get_ifo_data_generator(
            time_interval = O3,
            data_labels = ["noise", "glitches"],
            ifo = "L1",
            sample_rate_hertz = sample_rate_hertz,
            example_duration_seconds = example_duration_seconds,
            max_segment_size = 3600,
            max_num_examples = num_far_tests,
            num_examples_per_batch = batch_size,
            order = "random",
            apply_whitening = True,
            return_keys = ["data"]
        ).map(lambda x: x["data"])
        
        cbc_ds = cbc_ds \
            .batch(batch_size) \
            .prefetch(tf.data.experimental.AUTOTUNE) \
            .with_options(options)
        
        noise_ds = noise_ds \
            .prefetch(tf.data.experimental.AUTOTUNE) \
            .with_options(options)
        
        balanced_dataset = cbc_ds.concatenate(noise_ds \
            .take(tf.data.experimental.cardinality(cbc_ds).numpy())) \
            .prefetch(tf.data.experimental.AUTOTUNE) \
            .with_options(options)
        
        roc_data = {}
        for model_name in model_names:
            
            logging.info(f"Loading model {model_name}...")
            model = tf.keras.models.load_model(f"{skywarp_data_directory}/models/{model_name}")
            logging.info("Done.")
                
            """         
            logging.info(f"Calculate {model_name} ROC data...")
            fpr, tpr, roc_auc = calculate_roc_data(model, balanced_dataset)
            roc_data = {'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc}
            logging.info("Done.")

            logging.info(f"Calculating {model_name} efficiency scores...")
            path_suffix = f"{skywarp_data_directory}/datasets/cbc"
            efficiency_scores = calculate_efficiency_scores(model, path_suffix, batch_size, options)
            logging.info("Done.")
            """
                        
            logging.info(f"Calculating {model_name} FAR scores...")
            far_scores = calculate_far_scores(model, real_noise_ds, num_examples=num_far_tests)
            logging.info("Done.")
            
            quit()
            
            logging.info(f"Saving {model_name} validation data...")
            save_data_to_hdf5(model_name, "white_noise", roc_data, efficiency_scores, far_scores)
            logging.info("Done.")

            # Force garbage collection
            gc.collect()
                
        # Create TensorFlow dataset from the generator
        noise_ds = tf.data.Dataset.from_generator(
            generator=lambda: gaussian_noise_generator(num_samples=num_samples),
            output_signature=(
                tf.TensorSpec(shape=(num_samples,), dtype=tf.float16),
                tf.TensorSpec(shape=(), dtype=tf.float32),
            )
        ).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE).with_options(options)
        
        balanced_dataset = cbc_ds.concatenate(noise_ds \
            .take(tf.data.experimental.cardinality(cbc_ds).numpy())) \
            .prefetch(tf.data.experimental.AUTOTUNE) \
            .with_options(options)
        
        for model_name in model_names:
            
            logging.info(f"Loading model {model_name}...")
            model = tf.keras.models.load_model(f"{skywarp_data_directory}/models/{model_name}")
            logging.info("Done.")
                        
            logging.info(f"Calculate {model_name} ROC data...")
            fpr, tpr, roc_auc = calculate_roc_data(model, balanced_dataset)
            roc_data = {'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc}
            logging.info("Done.")

            logging.info(f"Calculating {model_name} efficiency scores...")
            path_suffix = f"{skywarp_data_directory}/datasets/cbc"
            efficiency_scores = calculate_efficiency_scores(model, path_suffix, batch_size, options)
            logging.info("Done.")
                        
            logging.info(f"Calculating {model_name} FAR scores...")
            far_scores = calculate_far_scores(model, noise_ds, num_examples=num_far_tests)
            logging.info("Done.")
            
            logging.info(f"Saving {model_name} validation data...")
            save_data_to_hdf5(model_name, "real_noise", roc_data, efficiency_scores, far_scores)
            logging.info("Done.")

            # Force garbage collection
            gc.collect()