import tensorflow as tf
from tqdm import tqdm
import tensorflow_datasets as tfds
from functools import partial

from tensorflow.keras import mixed_precision
from tensorflow.keras import backend as K

import numpy as np

from tensorflow import keras
from tensorflow.data import Dataset
from tensorflow.keras import layers

import matplotlib.pyplot as plt

from common_functions import *

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

strategy = setup_CUDA(True, "3,4,5,6,7")

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

def calculateEquivilentKernelSize(kernel_size, dilation_size):
	
	return int(kernel_size + (dilation_size - 1)*(kernel_size - 1))

def calculateConvOuputSize(input_size, kernel_size, stride_size, dilation_size):
	
	kernel_size = calculateEquivilentKernelSize(kernel_size, dilation_size)
	
	return int(((input_size - kernel_size) / stride_size ) + 1)

def get_element_shape(dataset):
    for element in dataset:
        return element['strain'].shape[1:]
    
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
    x = layers.Reshape((input_shape[-1], 1))(x)
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
    x = layers.Reshape((input_shape[-1], 1))(x)
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
        
    x = layers.Dense(2, activation="softmax", dtype=tf.float32)(x)
    
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
    
    inputs = keras.Input(shape=input_shape, name='strain')
    
    model_dim = num_heads * head_size
    
    if (res_head):
        x = build_resnet_head(input_shape, inputs)
    elif (conv_head):
        x = build_cnn_head(input_shape, inputs)
    else: 
        # Segmenting
        x = layers.Reshape((512, 16))(inputs)
            
    if (num_transformer_blocks > 0):
        # Embedd to higher dimensionality to increase the size of the model    
        x = layers.Conv1D(filters=model_dim, kernel_size=1, padding='valid', activation='relu')(x)

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
    
def getInput(element):
    return (element['strain'], tf.cast(element['signal_present'], tf.float32))

def lr_scheduler(epoch, lr, warmup_epochs=15, decay_epochs=100, initial_lr=1e-6, base_lr=1e-3, min_lr=5e-5):
    if epoch <= warmup_epochs:
        pct = epoch / warmup_epochs
        return ((base_lr - initial_lr) * pct) + initial_lr

    if epoch > warmup_epochs and epoch < warmup_epochs+decay_epochs:
        pct = 1 - ((epoch - warmup_epochs) / decay_epochs)
        return ((base_lr - min_lr) * pct) + min_lr

    return min_lr

if __name__ == "__main__":
    
    data_directory_path = "../skywarp_data"
    data_dir = f"{data_directory_path}/skywarp_dataset_gaussian"

    validation_signal_paths = [f"{data_directory_path}/datasets/cbc_10_e"]
    validation_noise_paths  = [f"{data_directory_path}/datasets/noise_0_v"]

    model_config_large = dict(
        name = "skywarp_conv_attention_large",
        res_head = False,
        conv_head = True,
        head_size=32,
        num_heads=10,
        ff_dim=10,
        num_transformer_blocks=10,
        mlp_units=[1024],
        mlp_dropout=0.1,
        dropout=0.1
    )
    
    conv_regular = dict(
        name = "skywarp_conv_regular",
        res_head = False,
        conv_head = True,
        head_size=16,
        num_heads=8,
        ff_dim=8,
        num_transformer_blocks=0,
        mlp_units=[64],
        mlp_dropout=0.1,
        dropout=0.5
    )
    
    pure_attention_regular = dict(
        name = "skywarp_attention_regular",
        res_head = False,
        conv_head = True,
        head_size=16,
        num_heads=8,
        ff_dim=8,
        num_transformer_blocks=8,
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
        num_transformer_blocks=8,
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
    
    test_models = [conv_regular, pure_attention_regular, conv_attention_regular, conv_attention_single]
        
    training_config = dict(
        learning_rate=1e-4,
        patience=20,
        epochs=200,
        batch_size=32
    )

    # Load Dataset:
    train_dataset = tfds.load(
        "skywarp_dataset_gaussian",
        data_dir = f"{data_directory_path}/skywarp_dataset_gaussian"
    )['train'].batch(batch_size=training_config["batch_size"])
        
    # Split Dataset:
    signal_v_dataset = load_label_datasets(validation_signal_paths, 1)
    noise_v_dataset = load_label_datasets(validation_noise_paths, 0)
    
    validation_dataset = signal_v_dataset.concatenate(noise_v_dataset).batch(
            batch_size=training_config["batch_size"]
        )
    
    validation_dataset, test_dataset = split_test_train(
        validation_dataset, 0.5)
    
    # Get Signal Element Shape:
    input_shape = get_element_shape(train_dataset)

    with strategy.scope():
        # model = tf.keras.models.load_model(f"models/skywarp_regular_c_10_10.2")
        
        for model_config in test_models:
            
            model_name = model_config["name"]
            model_path = f"{data_directory_path}/models/{model_name}"
            
            model = build_conv_transformer(
                input_shape,
                model_config
            )
                    
            model.compile(
                loss="sparse_categorical_crossentropy",
                optimizer=keras.optimizers.Adam(
                    learning_rate=training_config["learning_rate"]),
                metrics=["sparse_categorical_accuracy"],
            )
            model.summary()
            
            callbacks = [
                keras.callbacks.EarlyStopping(
                    patience=training_config["patience"],
                    restore_best_weights=True),
                keras.callbacks.ModelCheckpoint(
                    model_path,
                    monitor="val_loss",
                    save_best_only=True,
                    save_freq="epoch", 
                )
            ]

            history = model.fit(
                train_dataset.map(getInput, num_parallel_calls=tf.data.AUTOTUNE),
                validation_data=validation_dataset,
                epochs=training_config["epochs"],
                batch_size=training_config["batch_size"],
                callbacks=callbacks
            )

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

            model.evaluate(test_dataset, verbose=1)