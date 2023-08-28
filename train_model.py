import tensorflow as tf
from tqdm import tqdm
import tensorflow_datasets as tfds
from functools import partial

from tensorflow.keras import mixed_precision
from tensorflow.keras import backend as K

import sys
import argparse

import numpy as np

from tensorflow import keras
from tensorflow.data import Dataset
from tensorflow.keras import layers

import matplotlib.pyplot as plt

from py_ml_tools.dataset import get_ifo_data_generator, O3
from py_ml_tools.setup   import setup_cuda, find_available_GPUs

from tensorflow.data.experimental import AutoShardPolicy

import os

from tensorflow.keras.callbacks import Callback

def residual_block(inputs, kernel_size, num_kernels, num_layers):
    x = inputs
    for i in range(num_layers):
        x = layers.Conv1D(num_kernels, kernel_size, padding = 'same')(x) 
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
    inputs = layers.Conv1D(num_kernels, 1)(inputs) 
    
    return x + inputs

def identity_block(inputs, kernel_size, num_kernels, num_layers):
    x = inputs
    for i in range(num_layers):
        x = layers.Conv1D(num_kernels, kernel_size, padding = 'same')(x) 
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
    return x + inputs

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    # Conv1D(filters=ff_dim, kernel_size=1, activation="relu")
    x = layers.Dense(ff_dim, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    # Conv1D((filters=inputs.shape[-1], kernel_size=1))
    x = layers.Dense(inputs.shape[-1])(x)
    x = layers.Dropout(dropout)(x)
    return x + res

def positional_enc(seq_len: int, model_dim: int) -> tf.Tensor:
    """
    Computes pre-determined postional encoding as in (Vaswani et al., 2017).
    """
    pos = np.arange(seq_len)[..., None]
    dim = np.arange(model_dim, step=2)

    frequencies = 1.0 / np.power(1000, (dim / model_dim))

    positional_encoding_table = np.zeros((seq_len, model_dim))
    positional_encoding_table[:, 0::2] = np.sin(pos * frequencies)
    positional_encoding_table[:, 1::2] = np.cos(pos * frequencies)

    return tf.cast(positional_encoding_table, tf.float16)

def residual_block(inputs, kernel_size, num_kernels, num_layers):
    
    x = inputs
    for i in range(num_layers):
        x = layers.Conv1D(num_kernels, kernel_size, padding = 'same')(x) 
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
    inputs = layers.Conv1D(num_kernels, 1)(inputs) 
    
    return x + inputs

def build_cnn_head(input_shape, x):
    x = layers.Reshape((input_shape, 1))(x)
    x = layers.Conv1D(64, 8, activation="relu", padding = "same")(x)
    x = layers.MaxPool1D(8)(x)
    x = layers.Conv1D(32, 8, activation="relu", padding = "same")(x)
    x = layers.Conv1D(32, 16, activation="relu", padding = "same")(x)
    x = layers.MaxPool1D(6)(x)
    x = layers.Conv1D(16, 16, activation="relu", padding = "same")(x)
    x = layers.Conv1D(16, 32, activation="relu", padding = "same")(x)
    x = layers.Conv1D(16, 32, activation="relu", padding = "same")(x)
    
    return x

def build_resnet_head(input_shape, x):
    x = layers.Reshape((input_shape, 1))(x)
    x = residual_block(x, 8, int(model_dim/4), 2)
    x = layers.MaxPool1D(8)(x) 
    x = residual_block(x, 8, int(model_dim/2), 2)
    x = layers.MaxPool1D(8)(x) 
    x = residual_block(x, 8, int(model_dim), 2)
    
    return x

def build_dense_tail(model_config, x):
    
    # Unpack dict:
    mlp_units = model_config["mlp_units"]
    mlp_dropout = model_config["mlp_dropout"]
    
    x = layers.Flatten()(x)
    x = tf.cast(x, dtype=tf.float32)
    
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu", dtype=tf.float32)(x)
        x = layers.Dropout(mlp_dropout)(x)
        
    x = layers.Dense(2, activation="softmax", dtype=tf.float32, name = "injection_masks")(x)
    
    return x

def build_conv_transformer(
    input_shape,
    config,
    res_net = False
):
    #Unpack config:
    head_size = config["head_size"]
    num_heads = config["num_heads"]
    ff_dim = config["ff_dim"]
    num_transformer_blocks = config["num_transformer_blocks"]
    mlp_units = config["mlp_units"]
    mlp_dropout = config["mlp_dropout"]
    dropout = config["dropout"]
    res_head = config["res_head"]
    conv_head = config["conv_head"]
    
    inputs = keras.Input(shape=input_shape, name='onsource')
    
    model_dim = num_heads * head_size
    
    if (res_head):
        x = build_resnet_head(input_shape, inputs)
        # Embedd to higher dimensionality to increase the size of the model    
        x = layers.Conv1D(filters=model_dim, kernel_size=1, padding='valid', activation='relu')(x)
    elif (conv_head):
        x = build_cnn_head(input_shape, inputs)
        # Embedd to higher dimensionality to increase the size of the model    
        x = layers.Conv1D(filters=model_dim, kernel_size=1, padding='valid', activation='relu')(x)
    else: 
        # Segmenting
        x = layers.Reshape((-1, model_dim))(inputs)
        #x = layers.Conv1D(filters=model_dim, kernel_size=16, activation="relu", padding = "same")(x)
        #x = layers.MaxPool1D(16)(x) 
            
    if (num_transformer_blocks > 0):
        
        # positional encoding
        seq_len = x.shape[1]
        positional_encoding = positional_enc(seq_len, model_dim)  # or model_dim=1

        x += positional_encoding[:x.shape[1]]
        x = layers.Dropout(dropout)(x)

        for _ in range(num_transformer_blocks):
            x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

        x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    
    outputs = build_dense_tail(model_config, x)
    return keras.Model(inputs, outputs)

def lr_scheduler(
        epoch, 
        lr, 
        warmup_epochs=15, 
        decay_epochs=100, 
        initial_lr=1e-6, 
        base_lr=1e-3,
        min_lr=5e-5
    ):
    
    if epoch <= warmup_epochs:
        pct = epoch / warmup_epochs
        return ((base_lr - initial_lr) * pct) + initial_lr

    if epoch > warmup_epochs and epoch < warmup_epochs+decay_epochs:
        pct = 1 - ((epoch - warmup_epochs) / decay_epochs)
        return ((base_lr - min_lr) * pct) + min_lr

    return min_lr
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="A simple argument parser")
    parser.add_argument('model_index', type=int, help="Model Index")

    args = parser.parse_args()
    
    model_index = args.model_index
    
    gpus = find_available_GPUs(10000, 1)
    strategy = setup_cuda(gpus, 8000, verbose = True)
            
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = AutoShardPolicy.DATA
    
    # Parameters:
    num_examples_per_batch = 32
    sample_rate_hertz = 2048.0
    onsource_duration_seconds = 1.0
    max_segment_duration_seconds = 3600.0
    data_directory_path = f"../skywarp_data_{model_index}"
    num_train_examples = 1000000
    num_test_examples = 10000
    num_validate_examples = 10000
    num_examples_per_batch = 32
    
    conv_regular = dict(
        name = "skywarp_conv_regular",
        res_head = False,
        conv_head = True,
        head_size=16,
        num_heads=8,
        ff_dim=8,
        num_transformer_blocks=0,
        mlp_units=[64],
        mlp_dropout=0.5,
        dropout=0.5
    )
    
    pure_attention_regular = dict(
        name = "skywarp_attention_regular",
        res_head = False,
        conv_head = False,
        head_size=16,
        num_heads=8,
        ff_dim=8,
        num_transformer_blocks=6,
        mlp_units=[64],
        mlp_dropout=0.5,
        dropout=0.5
    )
    
    conv_attention_regular = dict(
        name = "skywarp_conv_attention_regular",
        res_head = False,
        conv_head = True,
        head_size=16,
        num_heads=8,
        ff_dim=8,
        num_transformer_blocks=6,
        mlp_units=[64],
        mlp_dropout=0.5,
        dropout=0.5
    )
    
    conv_attention_single = dict(
        name = "skywarp_conv_attention_single",
        res_head = False,
        conv_head = True,
        head_size=16,
        num_heads=8,
        ff_dim=8,
        num_transformer_blocks=1,
        mlp_units=[64],
        mlp_dropout=0.5,
        dropout=0.5
    )
    
    test_models = [
        conv_regular, 
        pure_attention_regular, 
        conv_attention_regular, 
        conv_attention_single
    ]
    
    test_models = [
        conv_attention_regular.copy()
        for i in range(5)
    ]
    for i, model_config in enumerate(test_models):
        model_config.update(
            {
                "name" : f"skywarp_conv_attention_{i}_layers",
                "num_transformer_blocks" : 2*i + 4
            }
        )
    
    test_models = [test_models[model_index]]
            
    training_config = \
        dict(
            learning_rate=1e-4,
            patience=10,
            epochs=200,
            batch_size=num_examples_per_batch
        )

    # Load Dataset:
    injection_config = \
        {
            "type" : "cbc",
            "snr"  : \
            {
                "min_value" : 8.0, 
                "max_value" : 20.0, 
                "distribution_type": "uniform"
            },
            "injection_chance" : 0.5,
            "padding_seconds" : \
            {
                "front" : 0.3, 
                "back" : 0.0
            },
            "args" : 
            {
                "mass_1_msun" : \
                {
                    "min_value" : 5, 
                    "max_value": 95, 
                    "distribution_type": "uniform"
                },
                "mass_2_msun" : \
                {
                    "min_value" : 5,
                    "max_value": 95, 
                    "distribution_type": "uniform"
                },
                "sample_rate_hertz" : \
                {
                    "value" : sample_rate_hertz, 
                    "distribution_type": "constant"
                },
                "duration_seconds" : \
                {
                    "value" : onsource_duration_seconds, 
                    "distribution_type": "constant"
                },
                "inclination_radians" : \
                {
                    "min_value" : 0, 
                    "max_value": np.pi, 
                    "distribution_type": "uniform"
                },
                "distance_mpc" : \
                {
                    "value" : 1000, 
                    "distribution_type": "constant"
                },
                "reference_orbital_phase_in" : \
                {
                    "min_value" : 0, 
                    "max_value": 2.0*np.pi, 
                    "distribution_type": "uniform"
                },
                "ascending_node_longitude" : \
                {
                    "min_value" : 0, 
                    "max_value": np.pi, 
                    "distribution_type": "uniform"
                },
                "eccentricity" : \
                {
                    "min_value" : 0, 
                    "max_value": 0.1, 
                    "distribution_type": "uniform"
                },
                "mean_periastron_anomaly" : \
                {
                    "min_value" : 0, 
                    "max_value": 2*np.pi, 
                    "distribution_type": "uniform"
                },
                "spin_1_in" : \
                {
                    "min_value" : -0.5, 
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
    
    generator_args = {
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
        "seed" : 100,
        "apply_whitening" : True,
        "input_keys" : ["onsource"], 
        "output_keys" : ["injection_masks"],
        "save_segment_data" : True
    }
    
    train_dataset = get_ifo_data_generator(
        **generator_args
    ).with_options(options).take(num_train_examples//num_examples_per_batch)
    
    validation_config = injection_config.copy()
    validation_config.update({
            "snr": {
                "min_value" : 6.0, 
                "max_value" : 10.0, 
                "distribution_type": "uniform"
            }
    })
    generator_args.update({
        "injection_configs" : [validation_config],
        "seed" : 101
    })    
        
    test_dataset = get_ifo_data_generator(
        **generator_args
    ).with_options(options).take(num_test_examples//num_examples_per_batch)
    
    generator_args.update({"seed" : 102})
    
    validation_dataset = get_ifo_data_generator(
        **generator_args
    ).with_options(options).take(num_validate_examples//num_examples_per_batch)
    
    def transform_features_labels(features, labels):
        labels['injection_masks'] = labels['injection_masks'][0]
        return features, labels
    
    # Get Signal Element Shape:
    input_shape = int(np.ceil(onsource_duration_seconds*sample_rate_hertz))

    with strategy.scope():        
        for model_config in test_models:
            
            model_name = model_config["name"]
            model_path = f"{data_directory_path}/models/{model_name}"
            
            model = \
                build_conv_transformer(
                    input_shape,
                    model_config
                )
                    
            model.compile(
                loss="sparse_categorical_crossentropy",
                optimizer=keras.optimizers.Adam(
                    learning_rate=training_config["learning_rate"]
                ),
                metrics=["sparse_categorical_accuracy"],
            )
            model.summary()
            
            def curriculum(epoch):
                epoch += 1
                injection_configs[0].update(
                    {"snr": 
                        {
                        "min_value" : np.maximum(10.0, 35.0 - epoch*5.0), 
                        "max_value" : np.maximum(20.0, 35.0 - epoch*2.5), 
                        "distribution_type": "uniform"
                        }
                    }
                )
                                
                return get_ifo_data_generator(
                    time_interval = O3,
                    data_labels = ["noise", "glitches"],
                    ifo = "L1",
                    injection_configs = injection_configs,
                    sample_rate_hertz = sample_rate_hertz,
                    onsource_duration_seconds = onsource_duration_seconds,
                    max_segment_size = max_segment_duration_seconds,
                    num_examples_per_batch = num_examples_per_batch,
                    data_directory = data_directory_path,
                    order = "random",
                    seed = 102 + epoch,
                    apply_whitening = True,
                    input_keys = ["onsource"], 
                    output_keys = ["injection_masks"],
                    save_segment_data = True
                ).with_options(options).take(num_validate_examples//num_examples_per_batch)
            
            class ModifyDatasetCallback(Callback):
                def __init__(self, train_dataset_function):
                    super(ModifyDatasetCallback, self).__init__()
                    self.train_dataset_function = train_dataset_function

                def on_epoch_end(self, epoch, logs=None):
                    self.model.stop_training = True  # Stop training
                    new_dataset = self.train_dataset_function(epoch)  # Create a new dataset
                    
                    self.model.fit(
                        train_dataset.map(transform_features_labels),
                        initial_epoch = epoch +1,
                        verbose = 2,
                        validation_data=validation_dataset.map(transform_features_labels),
                        epochs=training_config["epochs"],
                        batch_size=training_config["batch_size"],
                        callbacks=callbacks
                    ) # Continue training with the new dataset
                    self.model.stop_training = False  # Allow normal training process to continue
                    
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=training_config["patience"],
                    restore_best_weights=True,
                    start_from_epoch=4
                ),
                keras.callbacks.ModelCheckpoint(
                    model_path,
                    monitor="val_loss",
                    save_best_only=True,
                    save_freq="epoch", 
                ),
                #ModifyDatasetCallback(
                 #   curriculum
                #)
            ]

            history = model.fit(
                train_dataset.map(transform_features_labels),
                validation_data=validation_dataset.map(transform_features_labels),
                verbose = 2,
                epochs=training_config["epochs"],
                batch_size=training_config["batch_size"],
                callbacks=callbacks
            )
            
            print(history)

            model.save(model_path)

            plt.figure()
            plt.plot(history.history['sparse_categorical_accuracy'])
            plt.plot(history.history['val_sparse_categorical_accuracy'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'validation'], loc='upper left')
            plt.savefig(f"{data_directory_path}/plots/accuracy_history_{model_name}")

            plt.figure()
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'validation'], loc='upper left')
            plt.savefig(f"{data_directory_path}/plots/loss_history_{model_name}")

            print(
                model.evaluate(test_dataset.map(transform_features_labels), verbose=1) 
            )