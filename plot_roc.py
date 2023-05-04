import os
import numpy as np
import tensorflow as tf
from bokeh.plotting import figure, output_file, save, show
from bokeh.models import HoverTool, ColumnDataSource

from itertools import cycle

from tqdm import tqdm

from common_functions import *
from tensorflow.keras import mixed_precision

import gc
from tensorflow.keras import backend as K

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
    for chunk_idx in tqdm(range(num_chunks)):
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
    x_list, y_true_list = zip(*[(x.numpy(), y.numpy()) for x, y in dataset])

    x_concat = np.concatenate(x_list, axis=0)
    y_true = np.concatenate(y_true_list, axis=0)

    y_scores = model.predict(x_concat)[:, 1]
        
    y_true_tensor = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_scores_tensor = tf.convert_to_tensor(y_scores, dtype=tf.float32)

    fpr, tpr, roc_auc = roc_curve_and_auc(y_true_tensor, y_scores_tensor)
    
    return fpr.numpy(), tpr.numpy(), roc_auc.numpy()

def balance_datasets(cbc_ds, noise_ds):
    cbc_size = tf.data.experimental.cardinality(cbc_ds).numpy()
    noise_size = tf.data.experimental.cardinality(noise_ds).numpy()

    min_size = min(cbc_size, noise_size)

    cbc_ds = cbc_ds.take(min_size)
    noise_ds = noise_ds.take(min_size)

    return cbc_ds.concatenate(noise_ds)


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def remove_redundant_points(x, y, epsilon):
    if len(x) != len(y):
        raise ValueError("Lengths of x and y arrays should be the same.")
        
    new_x, new_y = [x[0]], [y[0]]
    
    for i in range(1, len(x)):
        if abs(x[i] - new_x[-1]) > epsilon or abs(y[i] - new_y[-1]) > epsilon:
            new_x.append(x[i])
            new_y.append(y[i])
            
    return new_x, new_y

def plot_and_save_roc_curves(models, roc_data, output_path, epsilon=1e-6):
    p = figure(title="Receiver Operating Characteristic (ROC) Curves",
               x_axis_label='False Positive Rate',
               y_axis_label='True Positive Rate',
               width=800, height=600)

    p.line([0, 1], [0, 1], color='navy', line_width=2, line_dash='dashed', legend_label="Random (area = 0.5)")

    colors = cycle(['red', 'green', 'blue', 'purple', 'orange', 'brown', 'pink', 'cyan', 'magenta', 'yellow'])
    
    for color, (model_name, (fpr, tpr, roc_auc)) in zip(colors, roc_data.items()):
        reduced_fpr, reduced_tpr = remove_redundant_points(fpr, tpr, epsilon)
        source = ColumnDataSource(data=dict(x=reduced_fpr, y=reduced_tpr))
        line = p.line(x='x', y='y', source=source, color=color, width=2, legend_label=f'{model_name} (area = {roc_auc:.5f})')
        
        hover = HoverTool(tooltips=[("Series", f"{model_name}"),
                                     ("False Positive Rate", "$x{0.0000}"),
                                     ("True Positive Rate", "$y{0.0000}")],
                          renderers=[line])
        p.add_tools(hover)

    p.legend.location = "bottom_right"

    output_file(output_path)
    save(p)

def load_datasets():
    dataset_labels = [("./datasets/cbc", 1), ("./datasets/noise", 0)]
    datasets = {
        label: load_label_datasets(
            [f"{prefix}_{i}_v" for i in range(6)], label
        )
        for prefix, label in dataset_labels
    }
    return datasets[0], datasets[1]

def prepare_balanced_dataset(cbc_ds, noise_ds):
    balanced_dataset = balance_datasets(cbc_ds, noise_ds) \
        .batch(32) \
        .prefetch(tf.data.experimental.AUTOTUNE) \
        .cache('./cache/dataset_cache')
    return balanced_dataset

def process_single_model(model_name, balanced_dataset):
    # Load the pre-trained model
    model = tf.keras.models.load_model(f"./models/{model_name}")

    # Calculate the ROC data using the custom function
    fpr, tpr, roc_auc = calculate_roc_data(model, balanced_dataset)

    # Save ROC data in NPY format
    roc_data_path = f'./roc_data/{model_name}_roc_data.npy'
    create_directory(os.path.dirname(roc_data_path))
    np.save(roc_data_path, {'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc})

    return (fpr, tpr, roc_auc)

def process_models(models, balanced_dataset):
    roc_data = {}
    for idx, model_name in enumerate(models):
        roc_data[model_name] = process_single_model(model_name, balanced_dataset)

        # Clear the TensorFlow session
        K.clear_session()
        
        # Force garbage collection
        gc.collect()

    return roc_data

if __name__ == "__main__":
    
    models = [
        "skywarp_res_conv",
        "cnn_10_10", 
        "skywarp_regular_c_10_10.3", 
        "skywarp_conv_c_10_10"
    ]

    strategy = setup_CUDA(True, "1,2,3,4,5,6,7")
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    
    with strategy.scope():
        cbc_ds, noise_ds = load_datasets()

        balanced_dataset = prepare_balanced_dataset(cbc_ds, noise_ds)
        roc_data = process_models(models, balanced_dataset)

    # Plot and save the ROC curves
    output_path = "./roc_curves/comparison_roc_curve.html"
    create_directory(os.path.dirname(output_path))
    
    plot_and_save_roc_curves(models, roc_data, output_path)