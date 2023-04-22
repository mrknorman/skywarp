import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras import layers
import numpy as np
import os

tf.keras.backend.set_floatx('float16')

LABEL_TYPE = np.float32

def setup_CUDA(verbose, device_num):
		
	os.environ["CUDA_VISIBLE_DEVICES"] = str(device_num)
		
	gpus =  tf.config.list_logical_devices('GPU')
	strategy = tf.distribute.MirroredStrategy(gpus)

	physical_devices = tf.config.list_physical_devices('GPU')
	
	for device in physical_devices:	

		try:
			tf.config.experimental.set_memory_growth(device, True)
		except:
			# Invalid device or cannot modify virtual devices once initialized.
			pass
	
	tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

	if verbose:
		tf.config.list_physical_devices("GPU")
		
	return strategy

def load_datasets(paths):
    
    dataset = tf.data.experimental.load(paths[0])
    for path in paths[1:]:
        dataset = dataset.concatenate(dataset)
        
    return dataset

def add_labels(dataset, label):
    dataset_size = dataset.cardinality().numpy()

    labels = Dataset.from_tensor_slices(
        np.full(dataset_size, label, dtype=np.float32))
    
    return Dataset.zip((dataset, labels))
    
def load_label_datasets(paths, label):
    
    dataset = load_datasets(paths)
    return add_labels(dataset, label)

def load_noise_signal_datasets(noise_paths, signal_paths):
    
    noise  = load_label_datasets(noise_paths, 0)
    signal = load_label_datasets(signal_paths, 1)

    return signal.concatenate(noise)

def split_test_train(dataset, fraction):
    dataset_size = dataset.cardinality().numpy()
    test_size = fraction * dataset_size

    dataset = dataset.shuffle(dataset_size)
    test_dataset = dataset.take(test_size)
    train_dataset = dataset.skip(test_size)

    return test_dataset, train_dataset