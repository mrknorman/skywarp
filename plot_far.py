import numpy as np
import matplotlib.pyplot as plt

from common_functions import *
from matplotlib.collections import LineCollection

def pres_format(ax1):
    
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

def calcFAR(model_names, fars, restriction):
        
    total_num_seconds = 0
    
    score_thresholds = {}
    for name in model_names:
        far_scores        = np.load(f"./far_scores/{name}.npy")[::-1]
        total_num_seconds = len(far_scores)
        far_axis          = (np.arange(total_num_seconds)+1)/total_num_seconds
        
        score_thresholds[name] = {}
        
        for far in fars:
            idx = np.abs(far - far_axis).argmin()
            idx = np.abs(far_scores - far_scores[idx]).argmin()
            
            if (far_scores[idx] == 1):
                 idx = (len(far_scores) - 1)  - np.abs(far_scores[::-1] - far_scores[idx]).argmin()
            
            score_thresholds[name][far] = (far_axis[idx], far_scores[idx])
    
    return score_thresholds


def plotFAR(model_names, fars, restriction):
    
    fig,ax1=plt.subplots(figsize=(6,6))
    
    total_num_seconds = 0
    
    score_thresholds = {}
    for name in model_names:
        far_scores        = np.load(f"./far_scores/{name}.npy")[::-1]
        total_num_seconds = len(far_scores)
        far_axis          = (np.arange(total_num_seconds)+1)/total_num_seconds
        ax1.loglog(far_scores, far_axis, label = name)
        
        """
        print(name)
        print((np.sum(far_scores == 0))/len(far_scores))
        print((np.sum(far_scores > 0.5))/len(far_scores))
        print((np.sum(far_scores > 0.9))/len(far_scores))
        print((np.sum(far_scores > 0.99))/len(far_scores))
        print((np.sum(far_scores > 0.999))/len(far_scores))
        print((np.sum(far_scores == 1))/len(far_scores))

        print(far_scores[:10])
        print(far_scores[-10:])
        """
        
        score_thresholds[name] = {}
        
        for far in fars:
            idx = np.abs(far - far_axis).argmin()
            idx = np.abs(far_scores - far_scores[idx]).argmin()
            
            if (far_scores[idx] == 1):
                 idx = (len(far_scores) - 1)  - np.abs(far_scores[::-1] - far_scores[idx]).argmin()
            
            score_thresholds[name][far] = (far_axis[idx], far_scores[idx])
            
    y_ticks, y_tick_labels = \
        returnTimeLabels(total_num_seconds)
    
    #ax1.set_yticks(y_ticks)
    #ax1.set_yticklabels(y_tick_labels)
    if (restriction>0):ax1.set_xlim(restriction,1)
    ax1.set_xlabel("Score Threshold")
    ax1.set_ylabel("False Alarms per Second")    
    pres_format(ax1)
    
    plt.savefig("FAR_plot", transparent=True)
    
    return score_thresholds

def calcEff(model_names, threshold):
        
    acc = {}
    for name in model_names:
        scores = np.load(f"./eff_plot_scores/{name}.npy")
        
        acc[name] = np.zeros([len(thresholds[name].keys()), len(scores)])
        for i, far in enumerate(thresholds[name].keys()):
            
            threshold  = thresholds[name][far][1]     
            actual_far = thresholds[name][far][0]

            for j, score in enumerate(scores):
                score = score[:,1]
                if (threshold != 0):
                    total = np.sum(score >= threshold)
                else:
                    total = np.sum(score > threshold)
                
                acc[name][i][j] = total/len(score)
    
    return acc


def plotEff(model_names, threshold):
    
    fig,ax1=plt.subplots(figsize=(6,6))
    
    ax1.set_xlabel('SNR ')
    ax1.set_ylabel('Accuracy')

    for name in model_names:
        scores = np.load(f"./eff_plot_scores/{name}.npy")
        
        for far in thresholds[name].keys():
            
            threshold  = thresholds[name][far][1]     
            actual_far = thresholds[name][far][0]
            acc = []
            for score in scores:
                score = score[:,1]
                if (threshold != 0):
                    total = np.sum(score >= threshold)
                else:
                    total = np.sum(score > threshold)
                acc.append(total/len(score))

            plt.plot(np.linspace(0,10,10), acc, linewidth=1, label = f"{name}_{actual_far:.4e}")

    pres_format(ax1)
    
    plt.savefig(f"plot_eff_{str(far).replace('.', '')}.png", transparent=True)
    
if __name__ == "__main__":
    restriction = 0
    model_names = ["skywarp_regular_c_10_10.2", "skywarp_conv_c_10_10_2", "skywarp_conv_single"]
    
    fars = [[1/10], [1/100], [1/1000], [1/10000], [1/100000], [1/1000000]]
    
    for far in fars:
        thresholds = plotFAR(model_names, far, 0)
        plotEff(model_names, thresholds)
    
    fars = (np.arange(10000)+1)[::1000]/1E6
        
    thresholds = calcFAR(model_names, fars, restriction)
    acc = calcEff(model_names, thresholds)
    
    print(acc)