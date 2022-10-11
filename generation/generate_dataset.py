import sys
sys.path.append("./mly/")
from mly.datatools import DataSet
import numpy as np
import tensorflow as tf
import os


tf.keras.backend.set_floatx('float16')

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


setup_CUDA(True, "1")

def generateDataset(config):
    
    print("Generate dataset...")
        
    # Generate Dataset:
    dataset = DataSet.generator(
       duration         = config["duration"],
       fs               = config["fs"],
       size             = config["num_injections"],
       labels           = config["labels"],
       detectors        = config["detectors"],
       backgroundType   = config["backgroundType"],
       #Inection related
       injectionFolder  = config["injectionFolder"],
       injectionSNR     = config["injectionSNR"],
       # More Options
       differentSignals = config["differentSignals"],
       maxDuration      = config["maxDuration"],
       disposition      = config["disposition"],
       windowSize       = config["windowSize"],
       plugins          = config["plugins"]
    )
    
    print("Save dataset...")
    
    data_array = np.empty([len(dataset), dataset[0].shape[1]])
    for index, pod in enumerate(dataset):
        data_array[index] = pod.strain[0]
    
    data_array = data_array.astype(np.float16)
    tf_dataset = tf.data.Dataset.from_tensor_slices(data_array)
    tf.data.experimental.save(tf_dataset, config["dataset_name"])
    
if __name__ == "__main__":
    
    num_datasets = 315;
    
    for i in range(10):   
        # User Parameters:
        config = dict(
            detector_initials = ["H1"],
            duration          = 1,
            fs                = 8192,
            num_injections    = 81000,
            labels            = {'type' : 'signal'},
            detectors         = 'H',
            backgroundType    = 'optimal',
            #Injection related:
            injectionFolder   = './new_injections',
            injectionSNR      = 10,
            injectionCrop     = 0.25,
            # More options:
            differentSignals  = False,
            maxDuration       = None,
            disposition       = None,
            windowSize        = 16,
            plugins           = [],
            dataset_name      = f"cbc_10_%i"
        )

        generateDataset(config)
