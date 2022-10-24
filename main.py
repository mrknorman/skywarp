import tensorflow as tf
from tqdm import tqdm
import tensorflow_datasets as tfds


from tensorflow.keras import mixed_precision

import numpy as np

from tensorflow import keras
from tensorflow.data import Dataset
from tensorflow.keras import layers

import matplotlib.pyplot as plt

from common_functions import *

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

strategy = setup_CUDA(True, "0,1,3,4,5")

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)


def get_element_shape(dataset):
    for element in dataset:
        return element['strain'].shape[1:]

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


class Time2Vec(keras.layers.Layer):
    def __init__(self, kernel_size=1):
        super(Time2Vec, self).__init__(trainable=True, name='Time2VecLayer')
        self.k = kernel_size
    
    def build(self, input_shape):
        # trend
        self.wb = self.add_weight(name='wb',shape=(input_shape[1],),initializer='uniform',trainable=True)
        self.bb = self.add_weight(name='bb',shape=(input_shape[1],),initializer='uniform',trainable=True)
        # periodic
        self.wa = self.add_weight(name='wa',shape=(1, input_shape[1], self.k),initializer='uniform',trainable=True)
        self.ba = self.add_weight(name='ba',shape=(1, input_shape[1], self.k),initializer='uniform',trainable=True)
        super(Time2Vec, self).build(input_shape)
    
    def call(self, inputs, **kwargs):
        bias = self.wb * inputs + self.bb
        dp = K.dot(inputs, self.wa) + self.ba
        wgts = K.sin(dp) # or K.cos(.)

        ret = K.concatenate([K.expand_dims(bias, -1), wgts], -1)
        ret = K.reshape(ret, (-1, inputs.shape[1]*(self.k+1)))
        return ret
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1]*(self.k + 1))

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


def build_transformer(
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

    inputs = keras.Input(shape=input_shape, name='strain')

    # rescale "chunking"
    x = layers.Reshape((512, 16))(inputs)

    # positional encoding
    seq_len = 512  # 1024
    model_dim = num_heads * head_size
    positional_encoding = positional_enc(seq_len, model_dim)  # or model_dim=1

    # projection to increase the size of the model
    x = layers.Conv1D(model_dim, 1)(x) 
    x += positional_encoding[:x.shape[1]]
    x = layers.Dropout(dropout)(x)

    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(2, activation="softmax", name = 'signal_present')(x)
    return keras.Model(inputs, outputs)

def build_cnn(
    input_shape,
    config
):
    
    inputs = keras.Input(shape=input_shape)
    x = layers.Reshape((input_shape[-1], 1))(inputs)

    x = layers.Conv1D(64, 8, activation="elu")(x) 
    x = layers.MaxPool1D(8)(x) 
    x = layers.Conv1D(32, 8, activation="elu")(x) 
    x = layers.Conv1D(32, 16, activation="elu")(x) 
    x = layers.MaxPool1D(6)(x) 
    x = layers.Conv1D(16, 16, activation="elu")(x) 
    x = layers.Conv1D(16, 32, activation="elu")(x) 
    #x = layers.MaxPool1D(4)(x) 
    x = layers.Conv1D(16, 32, activation="elu")(x) 
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="elu")(x) 
    x = layers.Dropout(0.5)(x) 
    x = layers.Dense(64, activation="elu")(x) 
    x = layers.Dropout(0.5)(x) 
    outputs = layers.Dense(2, activation="softmax", kernel_regularizer='l2')(x) 
        
    return keras.Model(inputs, outputs)

def getInput(element):
    return (element['strain'], tf.cast(element['signal_present'], tf.float16))

if __name__ == "__main__":
#
    data_dir = "./skywarp_dataset"

    validation_signal_paths = ["datasets/cbc_10_e"]
    validation_noise_paths  = ["datasets/noise_0_v"]
    
    model_name = "cnn_10_10"
    model_path = f"models/{model_name}"

    model_config_large = dict(
        head_size=32,
        num_heads=10,
        ff_dim=10,
        num_transformer_blocks=10,
        mlp_units=[1024],
        mlp_dropout=0.1,
        dropout=0.1
    )
    
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
    train_dataset = tfds.load(
        "skywarp_dataset",
        data_dir = "skywarp_dataset"
    )['train'].batch(batch_size=training_config["batch_size"])
    
    for i, data in enumerate(train_dataset.map(getInput, num_parallel_calls=tf.data.AUTOTUNE)):
        print(data)
        if i > 10:
            break
            
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

        model =  build_transformer(
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
            # verbose=2,
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
        plt.savefig(f"accuracy_history_{model_name}")

        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(f"loss_history_{model_name}")

        model.evaluate(test_dataset, verbose=1)
