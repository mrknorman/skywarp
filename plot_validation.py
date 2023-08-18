import numpy as np

from py_ml_tools.validate import Validator, plot_efficiency_curves, plot_far_curves, plot_roc_curves
from pathlib import Path

if __name__ == "__main__":
    
    # User parameters:
    data_directory = Path("../skywarp_data_0/")
    fars = np.logspace(-1, -7, 500)
    model_names = [
        "skywarp_attention_regular", 
        "skywarp_conv_attention_regular", 
        "skywarp_conv_attention_single", 
        "skywarp_conv_regular"
    ]
    
    data = {}
    validators = []
    for model_name in model_names:
        validators.append(
            Validator.load(data_directory / f"{model_name}_validation_data.h5")
        )
        
    plot_efficiency_curves(
        validators, 
        fars, 
        data_directory / "efficiency_curves.html"
    )
    
    plot_far_curves(
        validators,
        data_directory / "far_curves.html"
    )
    
    plot_roc_curves(
        validators,
        data_directory / "roc_curves.html"
    )