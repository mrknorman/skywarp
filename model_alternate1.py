import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,2,4,5'

#tf.debugging.set_log_device_placement(True)
gpus =  tf.config.list_logical_devices('GPU')
strategy = tf.distribute.MirroredStrategy(gpus)

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
    x = layers.Dense(ff_dim, activation="relu")(x) # Conv1D(filters=ff_dim, kernel_size=1, activation="relu")
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(inputs.shape[-1])(x) # Conv1D((filters=inputs.shape[-1], kernel_size=1))
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
    
    print(positional_encoding_table.shape)
    
    return tf.cast(positional_encoding_table, tf.float32)


def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout,
    mlp_dropout,
):
    inputs = keras.Input(shape=input_shape)
    
    # rescale "chunking"
    x = layers.Reshape((512, 16))(inputs)
    #x = layers.AveragePooling1D(8)(x)
    
    # positional encoding
    seq_len = 512 # 1024
    model_dim = num_heads * head_size
    positional_encoding = positional_enc(seq_len, model_dim) # or model_dim=1 
    
    # projection to increase the size of the model
    x = layers.Dense(model_dim)(x) # was Conv1D
    x += positional_encoding[:x.shape[1]]
    x = layers.Dropout(dropout)(x)
    
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)

noise_x  = np.load("/home/michael.norman/dragon/tools/numpy_datasets/starscream_noise_1.npy")
noise_y  = np.zeros(noise_x.shape[0], dtype = np.int32)
signal_x = np.load("/home/michael.norman/dragon/tools/numpy_datasets/starscream_signal_1.npy")
signal_y = np.ones(signal_x.shape[0], dtype = np.int32)

x = np.concatenate((noise_x, signal_x))
y = np.concatenate((noise_y, signal_y))

idx = np.random.permutation(len(x))

x = x[idx]
y = y[idx]

x_train = x[:int(0.95*len(x))]
x_test  = x[int(0.95*len(x)):]

y_train = y[:int(0.95*len(x))]
y_test  = y[int(0.95*len(x)):]

n_classes = len(np.unique(y_train))

input_shape = x_train.shape[1:]
with strategy.scope():
    
    model = build_model(
        input_shape,
        head_size=8,
        num_heads=4, # increased from 4 
        ff_dim=4,
        num_transformer_blocks=6,
        mlp_units=[512],
        mlp_dropout=0.10,
        dropout=0.10,
    )

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        metrics=["sparse_categorical_accuracy"],
    )
    model.summary()

    callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]

    model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=200,
        batch_size=64,
        #verbose=2, 
        callbacks=callbacks,
    )

    model.evaluate(x_test, y_test, verbose=1)

    model.save("starscream2")
