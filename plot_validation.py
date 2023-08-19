from py_ml_tools.validate import Validator
from pathlib import Path

if __name__ == "__main__":
    
    # User parameters:
    data_directory = Path("../skywarp_data_0/")
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
        
    validators[0].plot(
        data_directory / "validation_plots.html",
        comparison_validators = validators[1:]
    )
        
    