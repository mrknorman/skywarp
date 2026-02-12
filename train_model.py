"""
train_model.py - Training script for gravitational wave detection models.

Updated to use Keras 3 with JAX backend (via gravyflow).
"""

import os
import sys
import argparse
import logging

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from functools import partial

# Import gravyflow FIRST - it sets KERAS_BACKEND=jax before keras loads
import gravyflow as gf

# Keras 3 imports (must come AFTER gravyflow to use JAX backend)
import keras
from keras import layers, ops


def binary_focal_loss(gamma=2.0, alpha=0.25, fp_penalty=1.0, signal_smoothing=0.1):
    """
    Combines three SOTA techniques into one loss function:
    1. Focal Loss: Focuses on hard examples (glitches/low SNR).
    2. Asymmetric Penalty: Punishes False Positives (Noise predicted as Signal) 
       'fp_penalty' times more than Missed Signals.
    3. One-Sided Smoothing: Smooths Signal labels (1 -> 0.9) to prevent overfitting,
       but keeps Noise labels at 0 to strictly minimize False Positives.
    """
    def loss_fn(y_true, y_pred):
        # 1. Apply One-Sided Label Smoothing manually
        # If y_true is 1, it becomes (1 - 0.1) = 0.9
        # If y_true is 0, it stays 0.
        # This is safer for Low-FP tasks than standard smoothing.
        y_true_smooth = ops.where(
            y_true > 0.5, 
            1.0 - signal_smoothing, 
            0.0
        )

        epsilon = keras.backend.epsilon()
        y_pred = ops.clip(y_pred, epsilon, 1.0 - epsilon)

        # 2. Calculate Cross Entropy terms
        # Log of the probability that it IS a signal
        ce_signal = -y_true_smooth * ops.log(y_pred)
        # Log of the probability that it IS noise
        ce_noise = -(1 - y_true_smooth) * ops.log(1 - y_pred)
        
        # 3. Calculate Focal Weights
        # If truth is Signal, weight = (1 - pred)^gamma
        weight_signal = alpha * ops.power(1 - y_pred, gamma)
        # If truth is Noise, weight = (pred)^gamma
        weight_noise = (1 - alpha) * ops.power(y_pred, gamma)
        
        # 4. Combine and Apply Asymmetric Penalty
        loss_signal = weight_signal * ce_signal
        
        # Apply penalty ONLY to the noise term (The False Positive Risk)
        loss_noise = weight_noise * ce_noise * fp_penalty 
        
        return ops.mean(loss_signal + loss_noise)

    return loss_fn


# =============================================================================
# Adapter to Process Labels for Keras
# =============================================================================

class InjectionMaskAdapterDataset(keras.utils.PyDataset):
    """
    Adapter to convert GravyflowDataset output dictionaries to simple tensors.
    
    GravyflowDataset returns:
        inputs: {"ONSOURCE": tensor, "OFFSOURCE": tensor}
        outputs: {"INJECTION_MASKS": tensor}
    
    Keras model expects:
        inputs: {"ONSOURCE": tensor, "OFFSOURCE": tensor}  (dict is fine for multi-input)
        outputs: tensor  (simple tensor, NOT a dict, for single output)
    """
    
    def __init__(self, dataset):
        super().__init__(workers=0)
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        features, labels = self.dataset[index]
        
        # Extract INJECTION_MASKS from output dict and convert to simple tensor
        # Shape is (num_generators, batch, 1) -> need (batch, 1)
        injection_masks = labels[gf.ReturnVariables.INJECTION_MASKS.name]
        
        # Handle the generator dimension if present
        if len(injection_masks.shape) == 3:
            # Take first generator's masks
            injection_masks = injection_masks[0]
        
        # Ensure shape is (batch, 1) and cast to float32
        injection_masks = np.asarray(injection_masks, dtype=np.float32)
        if len(injection_masks.shape) == 1:
            injection_masks = injection_masks[:, np.newaxis]
        
        # Return features dict (for multi-input model) and simple tensor (for single output)
        return features, injection_masks


def residual_block(inputs, kernel_size, num_kernels, num_layers):
    x = inputs
    for i in range(num_layers):
        x = layers.Conv1D(num_kernels, kernel_size, padding='same')(x) 
        x = layers.ReLU()(x)
        x = layers.BatchNormalization()(x)
        
    inputs = layers.Conv1D(num_kernels, 1)(inputs) 
    
    return x + inputs


def identity_block(inputs, kernel_size, num_kernels, num_layers):
    x = inputs
    for i in range(num_layers):
        x = layers.Conv1D(num_kernels, kernel_size, padding='same')(x) 
        x = layers.ReLU()(x)
        x = layers.BatchNormalization()(x)
        
    return x + inputs


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    """
    Modernized Transformer Encoder Block (ViT Style).
    Changes:
    1. Uses GELU activation (Standard for SOTA Transformers).
    2. Explicitly casts MLP projection to prevent bottlenecking if ff_dim is set wrong.
    """
    # Calculate the model dimension automatically
    model_dim = inputs.shape[-1]
    
    # --- Sub-layer 1: Multi-Head Self-Attention ---
    # Pre-Norm architecture (Norm -> Attention -> Add) is more stable than Post-Norm
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    
    x = layers.MultiHeadAttention(
        key_dim=head_size, 
        num_heads=num_heads, 
        dropout=dropout
    )(x, x)  # Self-Attention: Query=x, Value=x
    
    # Stochastic Depth could be added here for very deep models, 
    # but Dropout is fine for now.
    x = layers.Dropout(dropout)(x)
    
    # Residual Connection
    res = x + inputs

    # --- Sub-layer 2: Feed Forward Network (MLP) ---
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    
    # CRITICAL FIX: Ensure ff_dim is an expansion, not a bottleneck.
    # If the config passed a small number, we override it to 4x model_dim
    # standard ViT expansion ratio is 4.
    actual_ff_dim = max(ff_dim, 4 * model_dim) 
    
    # Expansion Layer (Wide)
    x = layers.Dense(actual_ff_dim, activation="gelu")(x) 
    x = layers.Dropout(dropout)(x)
    
    # Projection Layer (Narrow) - Project back to model_dim
    x = layers.Dense(model_dim)(x)
    x = layers.Dropout(dropout)(x)
    
    return x + res


def positional_enc(seq_len: int, model_dim: int):
    """
    Computes pre-determined postional encoding as in (Vaswani et al., 2017).
    """
    pos = np.arange(seq_len)[..., None]
    dim = np.arange(model_dim, step=2)

    frequencies = 1.0 / np.power(1000, (dim / model_dim))

    positional_encoding_table = np.zeros((seq_len, model_dim))
    positional_encoding_table[:, 0::2] = np.sin(pos * frequencies)
    positional_encoding_table[:, 1::2] = np.cos(pos * frequencies)

    return ops.cast(positional_encoding_table, "float16")


def build_cnn_head(input_shape, x):
    """
    Implements a ConvNeXt-1D architecture optimized for Gravitational Wave (GW) data.
    
    Why this is SOTA for GW:
    1. Large Kernels (7x7): Captures long-range dependencies (the "chirp" evolution) 
       better than standard 3x3 VGG filters.
    2. Depthwise Separable Convs: Reduces parameters allow for a deeper network.
    3. Inverted Bottleneck: Expands channels 4x internally to learn complex features, 
       then projects back, preserving information flow.
    """
    
    # --- Configuration for ConvNeXt-Tiny (Scaled for Audio/1D) ---
    # Depths: Number of blocks per stage. [3, 3, 9, 3] is standard "Tiny".
    # Dims: Channel widths. [96, 192, 384, 768] is standard. 
    # Adjusted slightly for GW memory constraints:
    depths = [3, 3, 9, 3] 
    dims = [64, 128, 256, 512] 
    
    # --- Helper: The ConvNeXt Block ---
    def convnext_block(x, dim):
        shortcut = x
        
        # 1. Depthwise Conv (Spatial mixing) - Large Receptive Field
        # We use kernel_size=7 to mimic the global view of Transformers
        x = layers.Conv1D(filters=dim, kernel_size=7, padding="same", groups=dim)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # 2. Pointwise Conv (Channel mixing) - Expand 4x
        # Inverted Bottleneck design
        x = layers.Dense(4 * dim)(x)  # Dense acts as Conv1D(kernel=1) here
        x = layers.Activation("gelu")(x)
        
        # 3. Pointwise Conv (Channel mixing) - Project back
        x = layers.Dense(dim)(x)
        
        # 4. Residual Connection
        # Using a simple scaling factor (LayerScale) is common in SOTA, 
        # but simple addition works well for this scale.
        x = layers.Add()([shortcut, x])
        return x

    # --- Helper: Downsampling Layer ---
    def downsample_layer(x, dim):
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        # Stride 2 reduces time resolution by half
        x = layers.Conv1D(filters=dim, kernel_size=2, strides=2)(x)
        return x

    # --- Main Architecture Flow ---
    
    # 1. Stem: "Patchify" the input
    # Standard ConvNeXt uses stride 4. For GW data at 2048Hz, this
    # reduces the sequence length immediately to save compute.
    x = layers.Reshape((input_shape, 1))(x)
    x = layers.Conv1D(dims[0], kernel_size=4, strides=4)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    # 2. Stages
    # Loop through the 4 stages defined in 'depths' and 'dims'
    for i in range(4):
        # Apply blocks
        for _ in range(depths[i]):
            x = convnext_block(x, dims[i])
            
        # Apply downsampling (except after the very last stage)
        if i < 3:
            x = downsample_layer(x, dims[i+1])
            
    return x


def build_resnet_head(input_shape, x, model_dim):
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
    
    x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    x = layers.Lambda(lambda t: ops.cast(t, dtype="float32"))(x)
    
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu", dtype="float32")(x)
        x = layers.Dropout(mlp_dropout)(x)
        
    x = layers.Dense(1, activation="sigmoid", dtype="float32")(x)
    
    return x


def build_conv_transformer(
    input_shape,
    offsource_shape,
    config,
    res_net=False
):
    # Unpack config:
    head_size = config["head_size"]
    num_heads = config["num_heads"]
    ff_dim = config["ff_dim"]
    num_transformer_blocks = config["num_transformer_blocks"]
    mlp_units = config["mlp_units"]
    mlp_dropout = config["mlp_dropout"]
    dropout = config["dropout"]
    res_head = config["res_head"]
    conv_head = config["conv_head"]
    
    inputs = keras.Input(shape=input_shape, name='ONSOURCE')
    offsource_input = keras.Input(shape=offsource_shape, name='OFFSOURCE')
    
    # Whitening
    sample_rate_hertz = config.get("sample_rate_hertz", 2048.0)
    onsource_duration_seconds = config.get("onsource_duration_seconds", 1.0)
    
    x = gf.Whiten(
        sample_rate_hertz=sample_rate_hertz,
        onsource_duration_seconds=onsource_duration_seconds
    )([inputs, offsource_input])
    
    whitened_length = int(np.ceil(onsource_duration_seconds * sample_rate_hertz))
    
    model_dim = num_heads * head_size
    
    if res_head:
        x = build_resnet_head(whitened_length, x, model_dim)  # Pass model_dim
        # Embed to higher dimensionality to increase the size of the model    
        x = layers.Conv1D(filters=model_dim, kernel_size=1, padding='valid', activation='gelu')(x)
    elif conv_head:
        x = build_cnn_head(whitened_length, x)
        # Embed to higher dimensionality to increase the size of the model    
        x = layers.Conv1D(filters=model_dim, kernel_size=1, padding='valid', activation='gelu')(x)
    else: 
        # Segmenting
        x = layers.Reshape((-1, model_dim))(x)  # Use x from whiten
            
    if num_transformer_blocks > 0:
        
        # positional encoding
        seq_len = x.shape[1]
        positional_encoding = positional_enc(seq_len, model_dim)

        x += positional_encoding[:x.shape[1]]
        x = layers.Dropout(dropout)(x)

        for _ in range(num_transformer_blocks):
            x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    outputs = build_dense_tail(config, x)
    return keras.Model(inputs=[inputs, offsource_input], outputs=outputs)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="A simple argument parser")
    parser.add_argument('model_index', type=int, help="Model Index")

    args = parser.parse_args()
    
    model_index = args.model_index
    
    # Set up environment (GPU selection and memory management)
    gf.env(memory_to_allocate_tf=8000)
    
    # Set mixed precision policy for Keras 3
    # Note: With JAX backend, mixed precision is handled through dtype policies
    keras.mixed_precision.set_global_policy('mixed_float16')
    
    # Parameters:
    num_examples_per_batch = 32
    sample_rate_hertz = 2048.0
    onsource_duration_seconds = 16.0
    max_segment_duration_seconds = 3600.0
    data_directory_path = f"../skywarp_data_{model_index}"
    num_train_examples = 1000000
    num_test_examples = 10000
    num_validate_examples = 10000
    num_examples_per_batch = 32
    
    conv_regular = dict(
        name="skywarp_conv_regular",
        res_head=False,
        conv_head=True,
        head_size=16,
        num_heads=8,
        ff_dim=8,
        num_transformer_blocks=0,
        mlp_units=[64],
        mlp_dropout=0.5,
        dropout=0.5
    )
    
    pure_attention_regular = dict(
        name="skywarp_attention_regular",
        res_head=False,
        conv_head=False,
        head_size=16,
        num_heads=8,
        ff_dim=8,
        num_transformer_blocks=6,
        mlp_units=[64],
        mlp_dropout=0.5,
        dropout=0.5
    )
    
    conv_attention_regular = dict(
        name="skywarp_conv_attention_regular",
        res_head=False,
        conv_head=True,
        head_size=32,
        num_heads=8,
        ff_dim=1024,
        num_transformer_blocks=6,
        mlp_units=[64],
        mlp_dropout=0.1,
        dropout=0.5
    )
    
    conv_attention_single = dict(
        name="skywarp_conv_attention_single",
        res_head=False,
        conv_head=True,
        head_size=32,
        num_heads=8,
        ff_dim=1024,
        num_transformer_blocks=1,
        mlp_units=[64],
        mlp_dropout=0.1,
        dropout=0.5
    )
    
    test_models = [
        conv_regular, 
        pure_attention_regular, 
        conv_attention_regular, 
        conv_attention_single
    ]
    
    test_models = [test_models[model_index]]
            
    training_config = dict(
        learning_rate=1e-4,
        patience=10,
        epochs=200,
        batch_size=num_examples_per_batch
    )

    # IFO Data Obtainer
    ifo_data_obtainer = gf.IFODataObtainer(
        observing_runs=gf.ObservingRun.O3,
        data_quality=gf.DataQuality.BEST,
        data_labels=[gf.DataLabel.NOISE, gf.DataLabel.GLITCHES],
        segment_order=gf.SegmentOrder.RANDOM,
        force_acquisition=True,
        cache_segments=False
    )

    # Noise Obtainer
    noise = gf.NoiseObtainer(
        ifo_data_obtainer=ifo_data_obtainer,
        noise_type=gf.NoiseType.REAL,
        ifos=gf.IFO.L1
    )

    # Scaling Method (SNR)
    scaling_method = gf.ScalingMethod(
        value=gf.Distribution(min_=4.0, max_=50.0, type_=gf.DistributionType.UNIFORM),
        type_=gf.ScalingTypes.SNR
    )

    # Waveform Generator Distributions
    mass_1_distribution = gf.Distribution(min_=5.0, max_=95.0, type_=gf.DistributionType.UNIFORM)
    mass_2_distribution = gf.Distribution(min_=5.0, max_=95.0, type_=gf.DistributionType.UNIFORM)
    inclination_distribution = gf.Distribution(min_=0.0, max_=np.pi, type_=gf.DistributionType.UNIFORM)
    
    # Waveform Generator - Updated from cuPhenomDGenerator to CBCGenerator
    phenom_d_generator = gf.CBCGenerator(
        mass_1_msun=mass_1_distribution,
        mass_2_msun=mass_2_distribution,
        inclination_radians=inclination_distribution,
        scaling_method=scaling_method,
        injection_chance=0.5
    )

    # Training Dataset (raw GravyflowDataset)
    train_dataset_raw = gf.Dataset(
        noise_obtainer=noise,
        waveform_generators=phenom_d_generator,
        sample_rate_hertz=sample_rate_hertz,
        onsource_duration_seconds=onsource_duration_seconds,
        num_examples_per_batch=num_examples_per_batch,
        input_variables=[gf.ReturnVariables.ONSOURCE, gf.ReturnVariables.OFFSOURCE],
        output_variables=[gf.ReturnVariables.INJECTION_MASKS]
    )

    # Validation Dataset
    # Create a new scaling method for validation with different SNR range
    validation_scaling_method = gf.ScalingMethod(
        value=gf.Distribution(min_=6.0, max_=10.0, type_=gf.DistributionType.UNIFORM),
        type_=gf.ScalingTypes.SNR
    )
    
    validation_generator = gf.CBCGenerator(
        mass_1_msun=mass_1_distribution,
        mass_2_msun=mass_2_distribution,
        inclination_radians=inclination_distribution,
        scaling_method=validation_scaling_method,
        injection_chance=0.5
    )

    validation_dataset_raw = gf.Dataset(
        noise_obtainer=noise,
        waveform_generators=validation_generator,
        seed=101,
        group="validate",
        sample_rate_hertz=sample_rate_hertz,
        onsource_duration_seconds=onsource_duration_seconds,
        num_examples_per_batch=num_examples_per_batch,
        input_variables=[gf.ReturnVariables.ONSOURCE, gf.ReturnVariables.OFFSOURCE],
        output_variables=[gf.ReturnVariables.INJECTION_MASKS]
    )

    # Test Dataset
    test_dataset_raw = gf.Dataset(
        noise_obtainer=noise,
        waveform_generators=phenom_d_generator,  # Use original generator or specific test config
        seed=102,
        group="test",
        sample_rate_hertz=sample_rate_hertz,
        onsource_duration_seconds=onsource_duration_seconds,
        num_examples_per_batch=num_examples_per_batch,
        input_variables=[gf.ReturnVariables.ONSOURCE, gf.ReturnVariables.OFFSOURCE],
        output_variables=[gf.ReturnVariables.INJECTION_MASKS]
    )
    
    # Get shapes from dataset (before wrapping)
    for input_example, _ in train_dataset_raw:
        input_shape = input_example["ONSOURCE"].shape[1:]
        offsource_shape = input_example["OFFSOURCE"].shape[1:]
        break  # Only need one sample
    
    # Wrap datasets with adapter to convert dict outputs to simple tensors
    train_dataset = InjectionMaskAdapterDataset(train_dataset_raw)
    validation_dataset = InjectionMaskAdapterDataset(validation_dataset_raw)
    test_dataset = InjectionMaskAdapterDataset(test_dataset_raw)

    for model_config in test_models:
        
        model_name = model_config["name"]
        model_path = f"{data_directory_path}/models/{model_name}.keras"
        
        model_config.update({
            "sample_rate_hertz": sample_rate_hertz,
            "onsource_duration_seconds": onsource_duration_seconds
        })
        
        model = build_conv_transformer(
            input_shape,
            offsource_shape,
            model_config
        )
                
        # Define the schedule
        total_steps = int(num_train_examples / num_examples_per_batch) * training_config["epochs"]
        warmup_steps = int(0.1 * total_steps)  # Warmup for 10% of training

        # Cosine Decay with Warmup
        learning_rate_fn = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=training_config["learning_rate"],
            decay_steps=total_steps,
            alpha=0.0,  # Minimum LR at the very end
            warmup_target=training_config["learning_rate"],
            warmup_steps=warmup_steps
        )

        # Pass this 'learning_rate_fn' into AdamW instead of a static number
        optimizer = keras.optimizers.AdamW(
            learning_rate=learning_rate_fn,
            weight_decay=1e-2
        )

        model.compile(
            loss=binary_focal_loss(gamma=2.0, alpha=0.25),
            optimizer=optimizer,
            metrics=[
                keras.metrics.BinaryAccuracy(name="accuracy"),
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="recall")
            ],
        )
        model.summary()
        
        def curriculum(epoch):
            epoch += 1
            
            # Update SNR
            min_snr = np.maximum(10.0, 35.0 - epoch * 5.0)
            max_snr = np.maximum(20.0, 35.0 - epoch * 2.5)
            
            new_scaling_method = gf.ScalingMethod(
                value=gf.Distribution(min_=min_snr, max_=max_snr, type_=gf.DistributionType.UNIFORM),
                type_=gf.ScalingTypes.SNR
            )
            
            new_generator = gf.CBCGenerator(
                mass_1_msun=mass_1_distribution,
                mass_2_msun=mass_2_distribution,
                inclination_radians=inclination_distribution,
                scaling_method=new_scaling_method,
                injection_chance=0.5
            )
            
            return InjectionMaskAdapterDataset(gf.Dataset(
                noise_obtainer=noise,
                waveform_generators=new_generator,
                seed=102 + epoch,
                sample_rate_hertz=sample_rate_hertz,
                onsource_duration_seconds=onsource_duration_seconds,
                num_examples_per_batch=num_examples_per_batch,
                input_variables=[gf.ReturnVariables.ONSOURCE, gf.ReturnVariables.OFFSOURCE],
                output_variables=[gf.ReturnVariables.INJECTION_MASKS],
                steps_per_epoch=num_validate_examples // num_examples_per_batch
            ))
        
        class ModifyDatasetCallback(keras.callbacks.Callback):
            def __init__(self, train_dataset_function):
                super(ModifyDatasetCallback, self).__init__()
                self.train_dataset_function = train_dataset_function

            def on_epoch_end(self, epoch, logs=None):
                self.model.stop_training = True  # Stop training
                new_dataset = self.train_dataset_function(epoch)  # Create a new dataset
                
                self.model.fit(
                    new_dataset,
                    initial_epoch=epoch + 1,
                    verbose=1,
                    validation_data=validation_dataset,
                    epochs=training_config["epochs"],
                    batch_size=training_config["batch_size"],
                    callbacks=callbacks
                )  # Continue training with the new dataset
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
            # ModifyDatasetCallback(curriculum)  # Uncomment to enable curriculum learning
        ]

        history = model.fit(
            train_dataset,
            validation_data=validation_dataset,
            verbose=1,
            epochs=training_config["epochs"],
            callbacks=callbacks
        )
        
        model.save(model_path)

        # Plot training history
        plt.figure()
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
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
            model.evaluate(test_dataset, verbose=1) 
        )