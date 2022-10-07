import tensorflow as tf

from tensorflow.keras import mixed_precision

import numpy as np

from tensorflow import keras
from tensorflow.data import Dataset
from tensorflow.keras import layers

import matplotlib.pyplot as plt

from common_functions import *

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

strategy = setup_CUDA(True, "1,2,3")

def get_element_shape(dataset):
    for element in dataset:
        return element[0].shape[1:]

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

    return tf.cast(positional_encoding_table, tf.float32)


def build_model(
    input_shape,
    config
):

    # Unpack dict:

    head_size = config["head_size"]
    num_heads = config["num_heads"]
    ff_dim = config["ff_dim"]
    num_transformer_blocks = config["num_transformer_blocks"]
    mlp_units = config["mlp_units"]
    mlp_dropout = config["mlp_dropout"]
    dropout = config["dropout"]

    inputs = keras.Input(shape=input_shape)

    # rescale "chunking"
    x = layers.Reshape((512, 16))(inputs)

    # positional encoding
    seq_len = 512  # 1024
    model_dim = num_heads * head_size
    positional_encoding = positional_enc(seq_len, model_dim)  # or model_dim=1

    # projection to increase the size of the model
    x = layers.Conv1D(model_dim, 1)(x)  # was Conv1D
    x += positional_encoding[:x.shape[1]]
    x = layers.Dropout(dropout)(x)

    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(2, activation="softmax")(x)
    return keras.Model(inputs, outputs)


if __name__ == "__main__":

    # User parameters:
    noise_paths = ["datasets/noise_1"]
    signal_paths = ["datasets/cbc_10"]
    
    validation_signal_paths = ["datasets/cbc_10_e", "datasets/cbc_9_e", "datasets/cbc_8_e", "datasets/cbc_7_e", "datasets/cbc_6_e"]
    validation_noise_paths  = ["datasets/noise_0_v"]
    
    model_path = "models/starscream_regular_c_10"

    validation_fraction = 0.05
    test_fraction = 0.1

    model_config = dict(
        head_size=16,
        num_heads=8,
        ff_dim=8,
        num_transformer_blocks=8,
        mlp_units=[512],
        mlp_dropout=0.1,
        dropout=0.1
    )

    training_config = dict(
        learning_rate=1e-4,
        patience=10,
        epochs=200,
        batch_size=32
    )

    # Load Dataset:
    train_dataset = load_noise_signal_datasets(
        noise_paths, signal_paths).batch(
        batch_size=training_config["batch_size"])

    # Split Dataset:
    signal_v_dataset = load_label_datasets(validation_signal_paths, 1)
    noise_v_dataset = load_label_datasets(validation_noise_paths, 0)
    
    noise_v_dataset = noise_v_dataset.take(len(signal_v_dataset))
    validation_dataset = signal_v_dataset.concatenate(noise_v_dataset).batch(
            batch_size=training_config["batch_size"]
        )
    
    validation_dataset, test_dataset = split_test_train(
        validation_dataset, 0.5)
    
    train_dataset = train_dataset.shuffle(len(train_dataset))

    # Get Signal Element Shape:
    input_shape = get_element_shape(train_dataset)

    with strategy.scope():

        model = build_model(
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
            train_dataset,
            validation_data=validation_dataset,
            epochs=training_config["epochs"],
            batch_size=training_config["batch_size"],
            # verbose=2,
            callbacks=callbacks
        )

        model.save(model_path)

        plt.figure()
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig("accuracy_history")

        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig("loss_history")

        model.evaluate(test_dataset, verbose=1)
