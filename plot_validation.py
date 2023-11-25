from pathlib import Path

from py_ml_tools.validate import Validator

if __name__ == "__main__":
    
    # User parameters:
    data_directory = Path("./validation_data")
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
        
    