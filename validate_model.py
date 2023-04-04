import tensorflow as tf
#tf.keras.backend.set_floatx('float16')

import numpy as np

from tensorflow import keras
from tensorflow.data import Dataset
from tensorflow.keras import layers

from tensorflow.keras import mixed_precision

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

import matplotlib.pyplot as plt

from tqdm import tqdm

from common_functions import *

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

strategy = setup_CUDA(True, "2,4,5")

def validate_model_scores(model, dataset):
    
    scores = np.empty([1, 2])
    for element in tqdm(dataset.batch(batch_size = 32)):
        scores = np.append(scores, model(element).numpy(), axis = 0)

    scores = scores[1:]
    
    return scores

def run_efficiency_plots(model, path_suffix, num_tests):
    
    acc = np.empty(num_tests)
    loss = np.empty(num_tests)
    
    for index in range(num_tests):
        
        path = [f"{path_suffix}_{index}_e"];
        dataset = load_label_datasets(path, 1);
        
        results = model.evaluate(dataset.batch(batch_size = 32), verbose=1)
        
        loss[index]= results[0]
        acc[index] = results[1]
        
    return acc, loss  

def run_efficiency_scores(model, path_suffix, num_tests):
    
    scores = []
<<<<<<< HEAD
    for index in range(10):
=======
    for index in np.linspace(0,10,21):
>>>>>>> 367c358eeef4ac984e599ea3fcbee421122690b5
        
        path = [f"{path_suffix}_{index}_e"];
        dataset = load_datasets(path);
        
        scores.append(validate_model_scores(model, dataset))
        
    return scores

def calculateFAR(model, datasets):
    
    scores = np.empty([1, 2])
    for dataset in datasets: 
        scores = np.append(scores, validate_model_scores(model, dataset))
    
    np.sort(scores)
    
def posterFormatGraph():
    
    fig = plt.figure()

    ax = fig.add_subplot(111)

    ax.set_xlabel('SNR ')
    ax.set_ylabel('Accuracy ')
    
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(5)

    ax.xaxis.label.set_color('white')        #setting up X-axis label color to yellow
    ax.yaxis.label.set_color('white')          #setting up Y-axis label color to blue

    ax.tick_params(axis='x', colors='white')    #setting up X-axis tick color to red
    ax.tick_params(axis='y', colors='white')  #setting up Y-axis tick color to black

    ax.spines['left'].set_color('white')        # setting up Y-axis tick color to red
    ax.spines['bottom'].set_color('white')         #setting up above X-axis tick color to red
    
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    
    return fig     

def binaryAccuracyFromOneHot(one_hot_score):

	return 0.5 + 0.5*(one_hot_score[1] - one_hot_score[0])

if __name__ == "__main__":
    # User parameters:
    noise_paths = ["datasets/noise_0_v", "datasets/noise_1_v", "datasets/noise_2_v", "datasets/noise_3_v", "datasets/noise_4_v", "datasets/noise_5_v", "datasets/noise_6_v", "datasets/noise_7_v", "datasets/noise_8_v", "datasets/noise_9_v"]
    
<<<<<<< HEAD
    model_name = "skywarp_conv_single"
=======
    model_name = "skywarp_res_conv"
>>>>>>> 367c358eeef4ac984e599ea3fcbee421122690b5

    model = tf.keras.models.load_model(f"./models/{model_name}")
    
    print("Signal")
    path_suffix = "./datasets/cbc"
    scores = run_efficiency_scores(model, path_suffix, 11)
    np.save(f"./eff_plot_scores/{model_name}", scores)
            
    scores = np.array([])
    for path in noise_paths:
        noise_ds = load_datasets([path])

        with strategy.scope():
            one_hot_scores = validate_model_scores(model, noise_ds)

        scores = np.append(scores, one_hot_scores[:,1])
        
    scores = np.sort(scores)[::1]
    np.save(f"./far_scores/{model_name}", scores)
    noise_ds = add_labels(noise_ds, 0)

    print("Noise")
    print(model.evaluate(noise_ds.batch(batch_size = 32), verbose=1, return_dict = True))




