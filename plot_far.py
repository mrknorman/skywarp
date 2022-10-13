import numpy as np
import matplotlib.pyplot as plt

from common_functions import *

def returnTimeLabels(total_num_seconds, seconds_in_time = None):
        
    if seconds_in_time == None:
        seconds_in_time = {
            315360000: "$Decade^{-1}$", 
            31536000: "$Year^{-1}$",
            2628288: "$Month^{-1}$", 
            604800: "$Week^{-1}$", 
            86400: "$Day^{-1}$", 
            3600: "$Hour^{-1}$", 
            60: "$Minute^{-1}$",
            1: "$Second^{-1}$"
        }
    
    y_ticks       = []
    y_tick_labels = []
    for time in seconds_in_time.keys():
        if time < total_num_seconds:
            y_ticks.append(1/time)
            y_tick_labels.append(seconds_in_time[time])
            
    return y_ticks, y_tick_labels

def plotFAR(model_names, restriction):
    
    fig,ax1=plt.subplots(figsize=(6,6))
    
    total_num_seconds = 0
    for name in model_names:
        far_scores        = np.load(f"./far_scores/{name}.npy")
        total_num_seconds = len(far_scores)
        far_axis          = 1.0/(np.arange(total_num_seconds)+1)
        ax1.loglog(far_scores, far_axis, label = name)

    y_ticks, y_tick_labels = \
        returnTimeLabels(total_num_seconds)

    plt.legend()
    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels(y_tick_labels)
    if (restriction>0):ax1.set_xlim(restriction,1)
    ax1.set_xlabel("Score")
    ax1.set_ylabel("FAR")
    plt.grid()
    plt.savefig("FAR_plot")
    
if __name__ == "__main__":
    restriction = 0
    model_names = ["scores_large", "scores_large_3", "scores_tiny", "scores_small", "scores_regular","scores_cnn"]
    
    plotFAR(model_names, restriction)