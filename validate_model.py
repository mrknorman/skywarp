import tensorflow as tf
#tf.keras.backend.set_floatx('float16')

import numpy as np

from tensorflow import keras
from tensorflow.data import Dataset
from tensorflow.keras import layers

import matplotlib.pyplot as plt

from tqdm import tqdm

from common_functions import *

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

setup_CUDA(True, "0")

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

def validate_model_scores(model, dataset):
    
    scores = np.empty([1, 2])
    for element in tqdm(dataset.batch(batch_size = 32)):
        scores = np.append(scores, model(element).numpy(), axis = 0)

    scores = scores[1:]
    
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
    noise_paths = ["datasets/noise_0_v"]

    model = tf.keras.models.load_model("./models/starscream_regular_c_10")
    
    """
    print("Signal")
    path_suffix = "./datasets/cbc"
    acc, loss = run_efficiency_plots(model, path_suffix, 11)
    
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

    plt.plot(acc, linewidth=5)
    
    plt.savefig("eff_acc_tiny_c.png", transparent=True)
    
    plt.figure()
    plt.plot(loss)
    plt.savefig("eff_loss_tiny_c.png")
    
    """

    noise_ds = load_datasets(noise_paths)    
    one_hot_scores = validate_model_scores(model, noise_ds)
    scores = one_hot_scores[:,0]
    scores  = np.sort(scores)[::-1]

    print(scores[-10:])

    print(len(scores[scores < 0.5])/len(scores))
    print(len(scores[scores < 0.1])/len(scores))
    print(len(scores[scores < 0.01])/len(scores))
    print(len(scores[scores < 0.001])/len(scores))
    
    plt.loglog(scores)
    plt.savefig("FAR_plot")
    
    quit()

    noise_ds = add_labels(noise_ds, 0)

    print("Noise")
    print(model.evaluate(noise_ds.batch(batch_size = 32), verbose=1, return_dict = True))


