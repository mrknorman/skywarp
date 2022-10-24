import numpy as np
import matplotlib.pyplot as plt

from common_functions import *
from matplotlib.collections import LineCollection

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
        far_scores        = np.load(f"./far_scores/{name}.npy")[::-1]
        total_num_seconds = len(far_scores)
        far_axis          = (np.arange(total_num_seconds)+1)/total_num_seconds
        ax1.loglog(far_scores, far_axis, label = name)

        print(far_scores[:10])
        print(far_scores[-10:])
    
    y_ticks, y_tick_labels = \
        returnTimeLabels(total_num_seconds)

    plt.legend()
    #ax1.set_yticks(y_ticks)
    #ax1.set_yticklabels(y_tick_labels)
    if (restriction>0):ax1.set_xlim(restriction,1)
    ax1.set_xlabel("Score")
    ax1.set_ylabel("Events per Second")
    plt.grid()
    plt.savefig("FAR_plot")
    
def plotEff(model_names):
    
    fig,ax1=plt.subplots(figsize=(6,6))
    
    ax1.set_xlabel('SNR ')
    ax1.set_ylabel('Accuracy ')

    for name in model_names:
        acc, loss = np.load(f"./eff_plot_scores/{name}.npy")
        plt.plot(acc, linewidth=3, label = name)
    
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(5)

    ax1.xaxis.label.set_color('white')        #setting up X-axis label color to yellow
    ax1.yaxis.label.set_color('white')          #setting up Y-axis label color to blue

    ax1.tick_params(axis='x', colors='white')    #setting up X-axis tick color to red
    ax1.tick_params(axis='y', colors='white')  #setting up Y-axis tick color to black

    ax1.spines['left'].set_color('white')        # setting up Y-axis tick color to red
    ax1.spines['bottom'].set_color('white')         #setting up above X-axis tick color to red
    
    ax1.spines.right.set_visible(False)
    ax1.spines.top.set_visible(False)
    
    plt.legend()

    
    plt.savefig("plot_eff.png", transparent=True)
    #plt.figure()
    #plt.plot(loss)
    #plt.savefig("eff_loss_tiny_c.png")
    
if __name__ == "__main__":
    restriction = 0
    model_names = [ "cnn_10_10", "skywarp_large_c_10_10", 'scores_regular', 'scores_small', 'scores_tiny',  'scores_large', 'scores_large_3', 'scores_cnn']
    
    #plotEff(model_names)
    plotFAR(model_names, 0.5)