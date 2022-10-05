"""
Implement functionality to plot various metrics, including:
- stability of training of different models: violin plot of various keys
- confusion matrix
- coefficients of importance from logistic regression model

Code copied and modified from: serverstudy project
""" 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import List
from matplotlib import ticker as mticker
from matplotlib.text import Text
from matplotlib.collections import QuadMesh
from pathlib import Path

def plot_violin_from_dict(metric_dict: dict, ylimits: float=None, log: bool=False, show_zero: bool=False, title: str=None, ax: plt.Axes=None, figsize: tuple=(10, 6), sort_by_mean: bool=False, save_path: Path=None) -> None:
    """
    Given a dictionary (metric_dict) of various metrics corresponding to a list of those metrics across different training samples, plot a violin plot of those metrics
    """
    #data_to_plot = [results['acc'],results['f1'],results['auc']]
    data_to_plot = []
    keys = []
    for key, lst in metric_dict.items():
        data_to_plot.append(lst)
        keys.append(key)
    n = len(data_to_plot[0])

    if sort_by_mean: # sort key, data_to_plot
        combined = list(zip(keys, data_to_plot))
        combined = sorted(combined, key=lambda tup: np.mean(tup[1])) # sort by mean of the data
        keys = [key for key, _ in combined]
        data_to_plot = [data for _, data in combined]

    if ax is None:
        # Create a figure instance
        fig = plt.figure(figsize=figsize, dpi=80)
        # Create an axes instance
        ax = fig.add_axes([0.05,0.05,0.95,0.95])
    
    if title is None:
        title = 'Stability N='+str(n)
    ax.set_title(title)          
    ax.set_ylabel('Metric value')
    ax.set_xlabel('Metrics')

    labels = keys

    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    #ax.set_xticklabels(labels, rotation='vertical')
    ax.set_xticklabels(labels, rotation=45, ha='right')
    #ax.set_xlim(0.25, len(labels) + 0.75)
    #ax.set_xticklabels(ax.get_xticks(), rotation = 45)

    if ylimits is not None:
        ax.set_ylim(ylimits[0], ylimits[1])

    # plt.yscale('log') doesnt work properly for small values (and have negatives!)

    if log is True:
        log_data = [[np.log10(d) for d in row] for row in data_to_plot]
        data_to_plot = log_data

    # Create the boxplot
    bp = ax.violinplot(data_to_plot, showmeans=True)

    #for pc in bp['bodies']:
    #    pc.set_facecolor(color)
    #    #pc.set_edgecolor('black')
    #    #pc.set_alpha(1)
    
    if show_zero is True: 
        ax.axhline(y=0, color='r', linestyle='-')
    
    if log is True:
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("$10^{{{x:.0f}}}$"))
        #ax.yaxis.set_ticks([np.log10(x) for p in range(-6,1) for x in np.linspace(10**p, 10**(p+1), 10)], minor=True)
    
    if save_path is not None:
        fig.savefig(save_path)

    if ax is None: 
        plt.show()
    return bp

def logistic_regression_plot_coeffs(labels: List[str], values: List[float], title: str):
    """
    Plot importance of different features based on logistic regression coefficients
    """
    plt.figure(figsize=(16, 6), dpi=80)
    results = pd.DataFrame()
    results["feature"] = labels #feature_data.columns
    results["importance"] = values 
    results = results.sort_values("importance")
    plt.title(title)
    plt.bar(results["feature"], results["importance"])
    plt.xticks(rotation=45, ha='right')
    plt.axhline(y=0, color='r', linestyle='-')
    plt.show() 

def plot_cross_correlation(corr: pd.DataFrame, save_path: Path=None) -> None:
    """
    Plot a correlation matrix with a given title
    """
    fig, ax = plt.subplots(figsize=(16, 8)) 
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    dropvals = np.zeros_like(corr)
    dropvals[np.triu_indices_from(dropvals)] = True
    sns.heatmap(corr, cmap = colormap, linewidths = .5, annot = True, fmt = ".2f", mask = dropvals)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_confusion_matrix(conf_matrix: pd.DataFrame, title: str='Confusion Matrix', labels: list=None, marginal_text_color: str='Black', **kwargs) -> plt.Axes:
    """
    Plot a confusion matrix with sums of rows and columns

    Args:
        conf_matrix: the confusion matrix to be plotted
        labels: labels of the categories, default to integer values starting from 0
        title: title of the plot
        marginal_text_color: color for the text of the Total row and columns
        **kwargs: keyword arguments for sns.heatmap, except for xticklabels, yticklabels, and vmax
    """
    vmax = np.max(conf_matrix)
    conf_matrix = np.concatenate((conf_matrix, conf_matrix.sum(axis=1, keepdims=True)), axis=1) # pad total of labels
    conf_matrix = np.concatenate((conf_matrix, conf_matrix.sum(axis=0, keepdims=True)), axis=0) # pad total of predictions 

    if labels is None:
        labels = [str(i) for i in range(len(conf_matrix) - 1)] + ['Total']
    else:
        labels = [l for l in labels] + ['Total']

    ax = sns.heatmap(
        conf_matrix, 
        vmax=vmax,
        xticklabels=labels,
        yticklabels=labels,
        **kwargs
    )
    ax.set_xlabel('Prediction')
    ax.set_ylabel('Actual')
    ax.set_title(title)

    # set total columns to white color and black text
    # code inspired by https://stackoverflow.com/questions/34298052/how-to-add-column-next-to-seaborn-heat-map
    quadmesh = ax.findobj(QuadMesh)[0]
    facecolors = quadmesh.get_facecolors()
    facecolors[np.arange(len(labels) - 1, len(labels) ** 2 - 1, len(labels))] = np.array([1, 1, 1, 1])
    facecolors[-len(labels):-1] = np.array([1, 1, 1, 1])

    quadmesh.set_color(facecolors)
    for text in ax.findobj(Text):
        text.set_color(marginal_text_color)

    return ax
