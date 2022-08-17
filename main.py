import tensorflow as tf
import numpy as np

from tensorflow import keras
from tensorflow.data import Dataset
from tensorflow.keras import layers

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

gpus = tf.config.list_logical_devices('GPU')
strategy = tf.distribute.MirroredStrategy(gpus)

def loadDatasets(noise_path, signal_path):
    noise = tf.data.experimental.load(noise_path)
    noise_labels = Dataset.from_tensor_slices(
        np.zeros(len(noise), dtype=np.float16))

    noise = Dataset.zip((noise, noise_labels))

    signal = tf.data.experimental.load(signal_path)
    signal_labels = Dataset.from_tensor_slices(
        np.ones(len(signal), dtype=np.float16))

    signal = Dataset.zip((signal, signal_labels))

    return noise.concatenate(signal)

def splitTestTrain(dataset, fraction):
    dataset_size = dataset.cardinality().numpy()
    test_size = test_fraction * dataset_size

    dataset = dataset.shuffle(dataset_size)
    test_dataset = dataset.take(test_size)
    train_dataset = dataset.skip(test_size)

    return test_dataset, train_dataset

def getElementShape(dataset):

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
    noise_path = "datasets/noise_1"
    signal_path = "datasets/cbc_snr_10"
    
    validation_fraction = 0.05
    test_fraction       = 0.1

    model_config = dict(
        head_size=8,
        num_heads=4,
        ff_dim=4,
        num_transformer_blocks=6,
        mlp_units=[512],
        mlp_dropout=0.10,
        dropout=0.1
    )

    training_config = dict(
        learning_rate = 1e-4,
        patience=10,
        epochs=200,
        batch_size=64
    )

    # Load Dataset:
    dataset = loadDatasets(noise_path, signal_path).batch(batch_size=32)

    # Split Dataset:
    test_dataset, train_dataset = splitTestTrain(dataset, test_fraction)
    validation_dataset, train_dataset = splitTestTrain(train_dataset, validation_fraction)

    # Get Signal Element Shape:
    input_shape = getElementShape(train_dataset)
    
    print(input_shape)
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
                restore_best_weights=True)]

        model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=training_config["epochs"],
            batch_size=training_config["batch_size"],
            # verbose=2,
            callbacks=callbacks
        )

        model.evaluate(test_dataset, verbose=1)
        model.save("starscream2")
