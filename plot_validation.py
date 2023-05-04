import numpy as np

from common_functions import *
from matplotlib.collections import LineCollection
import matplotlib

from bokeh.embed import components

from itertools import cycle
from bokeh.resources import INLINE
from bokeh.resources import Resources
from bokeh.embed import file_html

import h5py

from bokeh.plotting import figure, show, save
from bokeh.models import Legend, ColumnDataSource, LogAxis, Range1d, LogTicker, Slider, CustomJS, Dropdown
from bokeh.io import export_png
from bokeh.layouts import gridplot, column

from bokeh.io import output_file, save
from bokeh.models import HoverTool, Legend, LogAxis

# Load data from HDF5 file
def load_data_from_hdf5(model_name):
    with h5py.File(f'./{model_name}_data.h5', 'r') as h5f:
        # Load ROC data
        roc_data = {
            'fpr': h5f['roc_data']['fpr'][()],
            'tpr': h5f['roc_data']['tpr'][()],
            'roc_auc': h5f['roc_data'].attrs['roc_auc']
        }

        # Load efficiency scores
        efficiency_scores = []
        eff_group = h5f['efficiency_scores']
        for i in range(len(eff_group.keys())):
            efficiency_scores.append(eff_group[f'score_{i}'][()])

        # Load FAR scores
        far_scores = h5f['far_scores']['scores'][()]

    return roc_data, efficiency_scores, far_scores

def return_time_labels(total_num_seconds, seconds_in_time = None):
        
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

def calculate_far_score_thresholds(model_scores, model_data, fars, restriction):
        
    total_num_seconds = 0
    
    score_thresholds = {}
    
    for name in model_names:
        far_scores        = data[name]["far_scores"]
        far_scores        = np.sort(far_scores)[::-1]
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

def plot_far_curves(model_names, model_titles, model_data, fars, output_path):
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
    
    colors = cycle(['red', 'blue', 'green', 'orange'])

    for color, (name, title) in zip(colors, zip(model_names, model_titles)):
        far_scores = model_data[name]["far_scores"]
                
        far_scores = np.sort(far_scores)[::-1]
        total_num_seconds = len(far_scores)
        far_axis = (np.arange(total_num_seconds, dtype=float) + 1) / total_num_seconds
                
        # Find the indices where far_scores changes
        unique_indices = np.concatenate(([0], np.where(np.diff(far_scores))[0] + 1))

        downsampled_far_scores = far_scores[unique_indices]
        downsampled_far_axis = far_axis[unique_indices]
        
        source = ColumnDataSource(
            data=dict(x=downsampled_far_scores, y=downsampled_far_axis, name=[title]*len(downsampled_far_scores))
        )
        
        line = p.line("x", "y", source=source, line_color=color, legend_label=title)

    hover = HoverTool()
    hover.tooltips = tooltips
    p.add_tools(hover)
    
    p.legend.location = "top_right"
    p.legend.click_policy = "hide"
    p.legend.click_policy = "hide"

    output_file(output_path, title="FAR Plot (Logarithmic)")
    save(p)
        
def plot_efficiency_curves(model_names, model_titles, model_data, thresholds, output_path):
    plot_width = 800
    plot_height = 600
    colors = cycle(['red', 'blue', 'green', 'orange'])

    snr = np.linspace(0, 10, 21)
    p = figure(
        width=plot_width,
        height=plot_height,
        x_axis_label="SNR",
        y_axis_label="Accuracy"
    )

    # Set the initial plot title
    far_keys = list(thresholds[model_names[0]].keys())
    p.title.text = f'FAR: {far_keys[0]}'

    legend_items = []
    all_sources = {}
    acc_data = {}

    for name, title, color in zip(model_names, model_titles, colors):
        scores = model_data[name]["efficiency_scores"]
        acc_all_fars = []

        for far_index, far in enumerate(thresholds[model_names[0]].keys()):
            threshold = thresholds[name][far][1]
            actual_far = thresholds[name][far][0]
            acc = []

            for score in scores:
                score = score[:, 1]
                if threshold != 0:
                    total = np.sum(score >= threshold)
                else:
                    total = np.sum(score > threshold)
                acc.append(total / len(score))

            acc_all_fars.append(acc)

        acc_data[name] = acc_all_fars
        source = ColumnDataSource(data=dict(x=snr, y=acc_all_fars[0], name=[title] * len(snr)))
        all_sources[name] = source
        line = p.line(x='x', y='y', source=source, line_width=1, line_color=color)
        legend_items.append((title, [line]))

    legend = Legend(items=legend_items, location="bottom_right")
    p.add_layout(legend)
    p.legend.click_policy = "hide"

    hover = HoverTool()
    hover.tooltips = [("Name", "@name"), ("SNR", "@x"), ("Accuracy", "@y")]
    p.add_tools(hover)


    slider = Slider(start=0, end=len(thresholds[model_names[0]].keys()) - 1, value=0, step=1, title=f"FAR Index: {far_keys[0]}")
    slider.background = 'white'

    callback = CustomJS(args=dict(slider=slider, sources=all_sources, plot_title=p.title, acc_data=acc_data, thresholds=thresholds, model_names=model_names, far_keys=far_keys), code="""
        const far_index = slider.value;
        const far_value = far_keys[far_index];
        plot_title.text = 'FAR: ' + far_value;

        for (const key in sources) {
            if (sources.hasOwnProperty(key)) {
                const source = sources[key];
                source.data.y = acc_data[key][far_index];
                source.change.emit();
            }
        }
    """)             
    slider.js_on_change('value', callback)

    # Add a separate callback to update the slider's title
    slider_title_callback = CustomJS(args=dict(slider=slider, far_keys=far_keys), code="""
        const far_index = slider.value;
        const far_value = far_keys[far_index];
        slider.title = 'FAR Index: ' + far_value;
    """)
    slider.js_on_change('value', slider_title_callback)

    layout = column(slider, p)
    
    output_file(output_path)
    save(layout) 
    
def remove_redundant_points(x, y, epsilon):
    if len(x) != len(y):
        raise ValueError("Lengths of x and y arrays should be the same.")
        
    new_x, new_y = [x[0]], [y[0]]
    
    for i in range(1, len(x)):
        if abs(x[i] - new_x[-1]) > epsilon or abs(y[i] - new_y[-1]) > epsilon:
            new_x.append(x[i])
            new_y.append(y[i])
            
    return new_x, new_y

def plot_roc_curves(model_names, data, output_path, epsilon=1e-6):
    p = figure(title="Receiver Operating Characteristic (ROC) Curves",
               x_axis_label='False Positive Rate',
               y_axis_label='True Positive Rate',
               width=800, height=600,
               x_axis_type='log', x_range=[1e-6, 1], y_range=[0, 1])

    p.line([0, 1], [0, 1], color='navy', line_width=2, line_dash='dashed', legend_label="Random (area = 0.5)")

    colors = cycle(['red', 'green', 'blue', 'purple', 'orange', 'brown', 'pink', 'cyan', 'magenta', 'yellow'])
    
    
    for color, model_name in zip(colors, model_names):
        
        roc_data = data[model_name]["roc_data"]

        fpr = roc_data["fpr"]
        tpr = roc_data["tpr"]
        roc_auc = roc_data["roc_auc"]
        
        reduced_fpr, reduced_tpr = remove_redundant_points(fpr, tpr, epsilon)
        source = ColumnDataSource(data=dict(x=reduced_fpr, y=reduced_tpr))
        line = p.line(x='x', y='y', source=source, color=color, width=2, legend_label=f'{model_name} (area = {roc_auc:.5f})')
        
        hover = HoverTool(tooltips=[("Series", f"{model_name}"),
                                     ("False Positive Rate", "$x{0.0000}"),
                                     ("True Positive Rate", "$y{0.0000}")],
                          renderers=[line])
        p.add_tools(hover)

    p.legend.location = "bottom_right"

    output_file(output_path)
    save(p)

if __name__ == "__main__":
    restriction = 0
    data_directory = "../skywarp_data/"
    model_names = [
        "skywarp_attention_regular", 
        "skywarp_conv_attention_regular", 
        "skywarp_conv_attention_single", 
        "skywarp_conv_regular"
    ]
    model_titles = [
        "Pure Attention Architecture", 
        "Convolotional Attention Architecture", 
        "Convolotional Single Attention Architecture", 
        "Convoloutional Architecture"
    ]

    fars = np.logspace(-1, -7, 500)
    
    data = {}
    for model in model_names:
        roc_data, efficiency_scores, far_scores = load_data_from_hdf5(f"{data_directory}/{model}")
        data[model] = {"roc_data": roc_data, "efficiency_scores" : efficiency_scores, "far_scores": far_scores}
        
    # Calculate thresholds
    thresholds = calculate_far_score_thresholds(model_names, data, fars, restriction)
    
    output_path = "../skywarp_data/efficiency_curves.html"
    plot_efficiency_curves(model_names, model_titles, data, thresholds, output_path)
    
    output_path = "../skywarp_data/far_curves.html"
    plot_far_curves(model_names, model_titles, data, fars, output_path)
    
    output_path = "../skywarp_data/roc_curves.html"
    plot_roc_curves(model_names, data, output_path, epsilon=1e-6)