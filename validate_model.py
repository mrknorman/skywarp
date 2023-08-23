import tensorflow as tf
import numpy as np
import logging

from tensorflow import keras
from tensorflow.data import Dataset
from tensorflow.keras import mixed_precision, layers
from tensorflow.keras import backend as K
from tensorflow.data.experimental import AutoShardPolicy

from py_ml_tools.dataset  import get_ifo_data, O3, get_ifo_data_generator
from py_ml_tools.setup    import setup_cuda, find_available_GPUs
from py_ml_tools.validate import Validator

from pathlib import Path
            
if __name__ == "__main__":
    
    # User parameters:
    skywarp_data_directory       = Path("../skywarp_data_0/")
    num_examples_per_batch       = 32
    sample_rate_hertz            = 2048.0
    max_segment_duration_seconds = 1024.0
    onsource_duration_seconds    = 1.0
    
    efficiency_config = \
        {
            "max_snr" : 20.0, 
            "num_snr_steps" : 11, 
            "num_examples_per_snr_step" : 8192//8
        }
    far_config = \
        {
            "num_examples" : 1.0E3
        }
    roc_config : dict = \
        {
            "num_examples" : 1.0E3,
            "snr_ranges" :  [
                (8.0, 20.0),
                8.0,
                10.0,
                12.0
            ]
        }
        
    gpus = find_available_GPUs(10000, 1)
    strategy = setup_cuda(gpus, 8000, verbose = True)
    
    model_names = [
        "skywarp_conv_attention_regular", 
        "skywarp_attention_regular", 
        "skywarp_conv_attention_single", 
        "skywarp_conv_regular"
    ]
    
    # Load datasets:
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    
    with strategy.scope():
        
        logging.basicConfig(level=logging.INFO)
        
        # Load Dataset:
        injection_config = \
            {
                "type" : "cbc",
                "snr"  : {
                    "min_value" : 8, 
                    "max_value" : 20, 
                    "distribution_type": "uniform"
                },
                "injection_chance" : 1.0,
                "padding_seconds" : {"front" : 0.3, "back" : 0.0},
                "args" : {
                    "mass_1_msun" : \
                        {"min_value" : 5, 
                         "max_value": 95, 
                         "distribution_type": 
                         "uniform"},
                    "mass_2_msun" : \
                        {"min_value" : 5, 
                         "max_value": 95, 
                         "distribution_type": "uniform"
                        },
                    "sample_rate_hertz" : \
                        {"value" : sample_rate_hertz,
                         "distribution_type": "constant"
                        },
                    "duration_seconds" : \
                        {"value" : onsource_duration_seconds, 
                         "distribution_type": "constant"
                        },
                    "inclination_radians" : \
                        {"min_value" : 0, 
                         "max_value": np.pi, 
                         "distribution_type": "uniform"
                        },
                    "distance_mpc" : \
                        {"value" : 1000, 
                         "distribution_type": "constant"
                        },
                    "reference_orbital_phase_in" : \
                        {"min_value" : 0, 
                         "max_value": 2*np.pi, 
                         "distribution_type": "uniform"
                        },
                    "ascending_node_longitude" : \
                        {"min_value" : 0, 
                         "max_value": np.pi,
                         "distribution_type": "uniform"
                        },
                    "eccentricity" : \
                        {
                        "min_value" : 0, "max_value": 1.0, 
                        "distribution_type": "uniform"
                        },
                    "mean_periastron_anomaly" : \
                        {"min_value" : 0, 
                         "max_value":  2*np.pi, 
                         "distribution_type": "uniform"
                        },
                    "spin_1_in" : \
                        {"min_value" : -0.5
                         "max_value": 0.5
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
        
        generator_args = {
            "time_interval" : O3,
            "data_labels" : ["noise", "glitches"],
            "ifo" : "L1",
            "injection_configs" : [injection_config],
            "sample_rate_hertz" : sample_rate_hertz,
            "onsource_duration_seconds" : onsource_duration_seconds,
            "max_segment_size" : max_segment_duration_seconds,
            "num_examples_per_batch" : num_examples_per_batch,
            "data_directory" : skywarp_data_directory,
            "order" : "random",
            "seed" : 200,
            "apply_whitening" : True,
            "input_keys" : ["onsource"], 
            "output_keys" : ["injection_masks"],
            "save_segment_data" : True
        }
        
        validators = []
        for model_name in model_names:
            
            logging.info(f"Loading model {model_name}...")
            model = tf.keras.models.load_model(
                skywarp_data_directory / f"models/{model_name}"
            )
            logging.info("Done.")
                        
            # Validate model:
            validator = \
                Validator.validate(
                    model, 
                    model_name,
                    generator_args,
                    num_examples_per_batch,
                    efficiency_config,
                    far_config,
                    roc_config
                )
            
            # Save validation data:
            validator.save(
                skywarp_data_directory / f"{model_name}_validation_data.h5", 
            )
            
            # Plot validation data:
            validator.plot(
                skywarp_data_directory / f"{model_name}_validation_plots.html"
            )
            
            validators.append(validator)
            
    # Plot all model validation data comparison:            
    validators[0].plot(
        skywarp_data_directory / "validation_plots.html",
        comparison_validators = validators[1:]
    )
