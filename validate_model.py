import tensorflow as tf

import numpy as np
import logging

from tensorflow import keras
from tensorflow.data import Dataset
from tensorflow.keras import mixed_precision, layers
from tensorflow.keras import backend as K
from tensorflow.data.experimental import AutoShardPolicy
from functools import reduce

from py_ml_tools.dataset  import get_ifo_data, O3, get_ifo_data_generator
from py_ml_tools.setup    import setup_cuda, find_available_GPUs
from py_ml_tools.validate import calculate_efficiency_scores

import matplotlib.pyplot as plt

import h5py

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

def calculate_roc_data(model, dataset):
    # Use .map() to extract the true labels and model inputs
    x_dataset = dataset.map(lambda x, y: x)
    y_true_dataset = dataset.map(lambda x, y: tf.cast(y['injection_masks'][0], tf.int32))
        
    # Convert the true labels dataset to a tensor using reduce
    tensor_list = []
    for batch in y_true_dataset:
        tensor_list.append(batch)

    y_true = tf.concat(tensor_list, axis=0)

    # Get the model predictions
    y_scores = model.predict(x_dataset, verbose = 1)[:, 1]

    # Calculate the ROC curve and AUC
    fpr, tpr, roc_auc = roc_curve_and_auc(y_true, y_scores)

    return fpr.numpy(), tpr.numpy(), roc_auc.numpy()

def calculate_far_scores(model, noise_ds, num_examples_per_batch, num_examples=1E5):
    num_steps = int(num_examples // num_examples_per_batch)
    noise_ds = noise_ds.take(num_steps)
    
    # Predict the scores and get the second column ([:, 1])
    far_scores = model.predict(noise_ds, steps = num_steps, verbose=2)[:, 1]
    
    return far_scores
                            
if __name__ == "__main__":
    
    # User parameters:
    skywarp_data_directory       = "../skywarp_data_0/"
    num_examples_per_batch       = 32
    num_far_tests                = int(1E4)
    sample_rate_hertz            = 2048.0
    max_segment_duration_seconds = 2048.0
    onsource_duration_seconds    = 1.0
    
    data_directory_path = "./skywarp_data"
    
    gpus = find_available_GPUs(10000, 1)
    strategy = setup_cuda(gpus, 8000, verbose = True)
    
    model_names = [
        "skywarp_conv_attention_regular", 
        "skywarp_attention_regular", 
        "skywarp_conv_attention_single", 
        "skywarp_conv_regular"
    ]
    
    # Load datasets:
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = AutoShardPolicy.DATA
    
    with strategy.scope():
        
        logging.basicConfig(level=logging.INFO)
        
        # Load Dataset:
        injection_config = \
            {
                "type" : "cbc",
                "snr"  : {
                    "min_value" : 8, 
                    "max_value" : 20, 
                    "distribution_type": "uniform"
                },
                "injection_chance" : 1.0,
                "padding_seconds" : {"front" : 0.3, "back" : 0.0},
                "args" : {
                    "mass_1_msun" : \
                        {"min_value" : 5, 
                         "max_value": 95, 
                         "distribution_type": 
                         "uniform"},
                    "mass_2_msun" : \
                        {"min_value" : 5, 
                         "max_value": 95, 
                         "distribution_type": "uniform"
                        },
                    "sample_rate_hertz" : \
                        {"value" : sample_rate_hertz,
                         "distribution_type": "constant"
                        },
                    "duration_seconds" : \
                        {"value" : onsource_duration_seconds, 
                         "distribution_type": "constant"
                        },
                    "inclination_radians" : \
                        {"min_value" : 0, 
                         "max_value": np.pi, 
                         "distribution_type": "uniform"
                        },
                    "distance_mpc" : \
                        {"value" : 1000, 
                         "distribution_type": "constant"
                        },
                    "reference_orbital_phase_in" : \
                        {"min_value" : 0, 
                         "max_value": 2*np.pi, 
                         "distribution_type": "uniform"
                        },
                    "ascending_node_longitude" : \
                        {"min_value" : 0, 
                         "max_value": np.pi,
                         "distribution_type": "uniform"
                        },
                    "eccentricity" : \
                        {
                        "min_value" : 0, "max_value": 0.1, 
                        "distribution_type": "uniform"
                        },
                    "mean_periastron_anomaly" : \
                        {"min_value" : 0, 
                         "max_value": 2*np.pi, 
                         "distribution_type": "uniform"
                        },
                    "spin_1_in" : \
                        {"min_value" : -0.5, 
                         "max_value": 0.5, 
                         "distribution_type": "uniform"
                        },
                    "spin_2_in" : \
                        {
                        "min_value" : -0.5,
                        "max_value": 0.5, 
                        "distribution_type": "uniform"
                        }
                }
            }

        injection_configs = [injection_config]

        cbc_args = {
            "time_interval" : O3,
            "data_labels" : ["noise", "glitches"],
            "ifo" : "L1",
            "injection_configs" : injection_configs,
            "sample_rate_hertz" : sample_rate_hertz,
            "onsource_duration_seconds" : onsource_duration_seconds,
            "max_segment_size" : max_segment_duration_seconds,
            "num_examples_per_batch" : num_examples_per_batch,
            "data_directory" : data_directory_path,
            "order" : "random",
            "seed" : 200,
            "apply_whitening" : True,
            "input_keys" : ["onsource"], 
            "output_keys" : ["injection_masks"],
            "save_segment_data" : True
        }
        
        cbc_ds = get_ifo_data_generator(
            **cbc_args
        ).with_options(options)
        
        balanced_args = cbc_args.copy()

        balanced_config = injection_config.copy()
        balanced_config.update({"injection_chance": 0.5})
        balanced_args.update({
            "injection_configs" : [balanced_config],
        })
        
        balanced_dataset = get_ifo_data_generator(
            **balanced_args
        ).with_options(options)
        
        noise_args = cbc_args.copy()
        
        noise_args.update({
            "injection_configs" : [],
            "output_keys" : []
        })
            
        noise_ds = get_ifo_data_generator(
            **noise_args
        ).with_options(options)
        
        for model_name in model_names:
            
            logging.info(f"Loading model {model_name}...")
            model = tf.keras.models.load_model(
                f"{skywarp_data_directory}/models/{model_name}"
            )
            logging.info("Done.")
        
            roc_data = {}  

            logging.info(f"Calculate {model_name} ROC data...")
            
            fpr, tpr, roc_auc = calculate_roc_data(
                model, balanced_dataset.take(1000//num_examples_per_batch)
            )
            roc_data = {'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc}
            logging.info("Done.")
            
            logging.info(f"Calculating {model_name} efficiency scores...")
            path_suffix = f"{skywarp_data_directory}/datasets/cbc"
            efficiency_scores = \
                calculate_efficiency_scores(
                    model, 
                    cbc_args,
                    32,
                    10.0,
                    41,
                    8192
                )
            logging.info("Done.")

            logging.info(f"Calculating {model_name} FAR scores...")
            
            far_scores = \
                calculate_far_scores(
                    model, 
                    noise_ds, 
                    num_examples_per_batch, 
                    num_examples=num_far_tests
                )
            logging.info("Done.")

            logging.info(f"Saving {model_name} validation data...")
            save_data_to_hdf5(
                model_name, 
                "real_noise", 
                roc_data, 
                efficiency_scores, 
                far_scores
            )
            logging.info("Done.")
