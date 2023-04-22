import numpy as np
import matplotlib.pyplot as plt

from common_functions import *
from matplotlib.collections import LineCollection
import matplotlib

from bokeh.plotting import figure, show, save
from bokeh.models import Legend, ColumnDataSource, LogAxis, Range1d, LogTicker
from bokeh.io import export_png
from bokeh.layouts import gridplot

from bokeh.io import output_file, save
from bokeh.models import HoverTool, Legend, LogAxis

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
            
            score_thresholds[name][far] = (far, far_scores[idx])
            
            if (far_scores[idx] == 1):
                 score_thresholds[name][far] = (far, 1.1)
    
    return score_thresholds

def plotFAR(model_names, model_titles, fars, num_samples=100000):
    plot_width = 800
    plot_height = 600
    tooltips = [
        ("Name", "@name"),
        ("Score Threshold", "@x"),
        ("False Alarms per Second", "@y"),
    ]

    p = figure(
        width=plot_width,
        height=plot_height,
        x_axis_label="Score Threshold",
        y_axis_label="False Alarms per Second",
        tooltips=tooltips,
        x_axis_type="log",
        y_axis_type="log"
    )
    
    total_num_seconds = 0
    score_thresholds = {}
    colors = ['red', 'blue', 'green']

    for color, (name, title) in zip(colors, zip(model_names, model_titles)):
        far_scores = np.load(f"./far_scores/{name}.npy")[::-1]
        total_num_seconds = len(far_scores)
        far_axis = (np.arange(total_num_seconds, dtype=float) + 1) / total_num_seconds
        
        # Downsample far_scores and far_axis
        downsample_indices = np.linspace(
            0, total_num_seconds - 1, num_samples, dtype=int, endpoint=True
        )
                
        # Find the indices where far_scores changes
        unique_indices = np.concatenate(([0], np.where(np.diff(far_scores))[0] + 1))


        downsampled_far_scores = far_scores[unique_indices]
        downsampled_far_axis = far_axis[unique_indices]
        
        source = ColumnDataSource(
            data=dict(x=downsampled_far_scores, y=downsampled_far_axis, name=[title]*len(downsampled_far_scores))
        )
        
        line = p.line("x", "y", source=source, line_color=color, legend_label=title)

        score_thresholds[name] = {}
        for far in [float(f) for f in fars]:
            idx = np.abs(far - far_axis).argmin()
            idx = np.abs(far_scores - far_scores[idx]).argmin()

            if far_scores[idx] == 1:
                idx = (len(far_scores) - 1) - np.abs(
                    far_scores[::-1] - far_scores[idx]
                ).argmin()

            score_thresholds[name][far] = (far_axis[idx], far_scores[idx])

    hover = HoverTool()
    hover.tooltips = tooltips
    p.add_tools(hover)
    
    p.legend.location = "top_right"
    p.legend.click_policy = "hide"
    p.legend.click_policy = "hide"

    output_file("FAR_plot_bokeh_log.html", title="FAR Plot (Logarithmic)")
    save(p)

    return score_thresholds

def calcEff(model_names, threshold, scores):
        
    acc = {}
    for name in model_names:        
        acc[name] = np.zeros([len(thresholds[name].keys()), len(scores[name])])
        for i, far in enumerate(thresholds[name].keys()):
            
            threshold  = thresholds[name][far][1]     
            actual_far = thresholds[name][far][0]

            for j, score in enumerate(scores[name]):
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

            plt.plot(np.linspace(0,10,21), acc, linewidth=1, label = f"{name}_{actual_far:.4e}")

    pres_format(ax1)
    
    plt.savefig(f"./plots/plot_eff_{str(far).replace('.', '')}.png", transparent=True)

if __name__ == "__main__":
    restriction = 0
    model_names = ["cnn_10_10", "skywarp_regular_c_10_10.3", "skywarp_conv_c_10_10"]
    model_titles = ["Convoloutional Architecture", "Pure Attention Architecture", "Convolotional Attention Architecture"]

    fars = [1/10, 1/100, 1/1000, 1/10000, 1/100000, 1/1000000]

    # Calculate thresholds
    thresholds = calcFAR(model_names, fars, restriction)

    # Call plotFAR_bokeh with the correct arguments
    plotFAR(model_names, model_titles, fars)