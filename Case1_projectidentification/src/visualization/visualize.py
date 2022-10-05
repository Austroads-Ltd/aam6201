import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import src.visualization.plot_metric as plot_metric
import matplotlib.patches as mpatches
import warnings

from sklearn.metrics import roc_auc_score, classification_report
from collections import OrderedDict
from typing import List, Dict
from pathlib import Path

warnings.filterwarnings('ignore')

year_map_dict = {
    'Treatment within 1 year': 'Year 1',
    'Treatment between 1 to 3 years': 'Year 2 - 3',
    'Treatment between 3 to 5 years': 'Year 4 - 5',
    'Treatment between 5 to 10 years': 'Year 6 - 10',
    'Treatment between 10 to 30 years': 'Year 11 - 30' 
}

treatment_time_order = {
    'Treatment within 1 year': 0,
    'Treatment between 1 to 3 years': 1,
    'Treatment between 3 to 5 years': 2,
    'Treatment between 1 to 2 years': 1,
    'Treatment between 2 to 5 years': 2,
    'Treatment between 5 to 10 years': 3,
    'Treatment between 10 to 30 years': 4
}

treatment_type_order = {
    'Resurfacing_SS': 0,
    'Resurfacing_AC': 1,
    'Major Patching': 2,
    'Rehabilitation': 3,
    'Retexturing': 4,
    'Regulation': 5
}

treatment_colors = {
    'Resurfacing_SS': 'tab:blue',
    'Resurfacing_AC': 'tab:orange',
    'Major Patching': 'tab:green',
    'Rehabilitation': 'tab:red',
    'Retexturing': 'tab:brown',
    'Regulation': 'tab:purple',
}

### UTIL METHODS ###
def update_summary_dict(current: dict, y_pred: np.ndarray, y_true: pd.Series, probs: np.ndarray, multiclass_roc: str='raise', label_names: list=None) -> None:
    """
    Given a current dict of summary statistics, append the new statistics computed from the new predictions 
    
    Args:
        x: input
        y: true labels
        model: the machine learning model
    """
    summary_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0, labels=label_names) # classification statistics for each label and each type of average weighting
    auc = roc_auc_score(y_true, probs, average=None, multi_class=multiclass_roc)

    # dict of summary statistics and feature importance
    weighted_dict = summary_dict['weighted avg'] # classification reports for weighted average
    current['f1-score'].append(weighted_dict['f1-score'])
    current['precision'].append(weighted_dict['precision'])
    current['recall'].append(weighted_dict['recall'])
    current['accuracy'].append(summary_dict['accuracy'])
    current['auc'].append(auc)


def plot_metric_by_treatment_type(
        project_label: pd.DataFrame, 
        running_conf_mat_list: List[np.array], 
        suptitle: str=None, 
        save_path: str=None, 
        estimator_type: str=None,
        experiment_prefix: str='train',
        experiment_suffix: str=None,
        experiment_folder: Path=None,
        dataset_name: str='Dummy',
        **kwargs
    ):
    """plot total accuracy for each of type-treatment pair"""
    types = project_label.columns.get_level_values(0)
    treatments = project_label.columns.get_level_values(1)
    uniq_types = list(dict.fromkeys(types))
    uniq_types = sorted(uniq_types, key=lambda x: treatment_time_order[x])
    uniq_treatments = list(dict.fromkeys(treatments))
    uniq_treatments = sorted(uniq_treatments, key=lambda x: treatment_type_order[x])

    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(12, 16))
    axs = axs.ravel()
    x = np.arange(len(uniq_types))
    width = 0.1

    metric_dict = {}
    metric_names = ['accuracy', 'precision', 'recall', 'f_score']
    for treatment in uniq_treatments:
        metric_dict[treatment] = {}
        for metric in metric_names:
            metric_time_dict = OrderedDict()
            for time_horizon in uniq_types:
                metric_time_dict[time_horizon] = [np.nan, np.nan]
            metric_dict[treatment][metric] = metric_time_dict
        
    for i, (time_type, treatment) in enumerate(project_label.columns):
        running_conf_mat = np.array(running_conf_mat_list)[:, i, :, :]
        metric_dict[treatment]['accuracy'][time_type] = ((running_conf_mat[:, 0, 0] + running_conf_mat[:, 1, 1]) / running_conf_mat.sum(axis=(1,2)))
        metric_dict[treatment]['precision'][time_type] = (running_conf_mat[:, 1, 1] / running_conf_mat[:, :, 1].sum(axis=1))
        metric_dict[treatment]['recall'][time_type] = (running_conf_mat[:, 1, 1] / running_conf_mat[:, 1, :].sum(axis=1))
        metric_dict[treatment]['f_score'][time_type] = (2 / (1 / metric_dict[treatment]['precision'][time_type] + 1 / metric_dict[treatment]['recall'][time_type]))

    handles = []
    acc_bars = []
    for i, treatment in enumerate(uniq_treatments):
        acc_bars.append(axs[0].violinplot(list(metric_dict[treatment]['accuracy'].values()), positions=x+i*width, widths=width, showmeans=True))
        acc_bars.append(axs[1].violinplot(list(metric_dict[treatment]['precision'].values()), positions=x+i*width, widths=width, showmeans=True))
        acc_bars.append(axs[2].violinplot(list(metric_dict[treatment]['recall'].values()), positions=x+i*width, widths=width, showmeans=True))
        acc_bars.append(axs[3].violinplot(list(metric_dict[treatment]['f_score'].values()), positions=x+i*width, widths=width, showmeans=True))
        for violin_plots in acc_bars:
            for key, collection in violin_plots.items():
                if key == 'bodies':
                    for pc in collection:
                        pc.set_facecolor(treatment_colors[treatment])
                        pc.set_alpha(0.3)
                        pc.set_edgecolor(treatment_colors[treatment])
                else:
                    collection.set_edgecolor(treatment_colors[treatment])
        handles.append(mpatches.Patch(color=acc_bars[0]["bodies"][0].get_facecolor().flatten()))
        acc_bars = []

    for metric in range(len(metric_names)):
        axs[metric].set_ylabel(metric_names[metric])
        axs[metric].set_xticks(x + width * (len(uniq_treatments) - 1) / 2)
        axs[metric].set_xticklabels([year_map_dict[treatment_time] for treatment_time in uniq_types], rotation=10)
        axs[metric].legend(handles, uniq_treatments, bbox_to_anchor=(1, 1), loc="upper left", title='Treatment Category')
        axs[metric].grid(True)
        axs[metric].set_ylim((-0.1, 1.1))

    if suptitle is not None:
        fig.suptitle(suptitle)
    else:
        fig.suptitle(' '.join([experiment_prefix.capitalize(), dataset_name, estimator_type, 'Results', ' '.join(map(str.capitalize, experiment_suffix.split('_')))]))
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    elif save_path is None:
        save_path = f'{experiment_prefix}_{estimator_type}_{dataset_name}_violin_{experiment_suffix}.jpg'
        plt.savefig(experiment_folder / save_path)

    plt.show()   

def plot_confusion_matrix_by_treatment_type(
        project_label: pd.DataFrame, 
        running_conf_mat_list: List[np.array], 
        suptitle:str=None, figsize=(16, 12), 
        save_path: str=None, 
        estimator_type: str=None,
        experiment_prefix: str='train',
        experiment_suffix: str=None,
        experiment_folder: Path=None,
        dataset_name: str='Dummy',
        **kwargs
    ):
    types = project_label.columns.get_level_values(0)
    treatments = project_label.columns.get_level_values(1)
    uniq_types = list(dict.fromkeys(types))
    uniq_types = sorted(uniq_types, key=lambda x: treatment_time_order[x])
    uniq_treatments = list(dict.fromkeys(treatments))
    uniq_treatments = sorted(uniq_treatments, key=lambda x: treatment_type_order[x])
    fig, axs = plt.subplots(nrows=len(uniq_treatments), ncols=len(uniq_types), figsize=(len(uniq_types) * 3, len(uniq_treatments) * 3 + 2))
    selected = np.zeros((len(uniq_treatments), len(uniq_types)))

    for i, (time_type, treatment) in enumerate(project_label.columns):
        conf_mat = np.array(running_conf_mat_list)[:, i, :, :].sum(axis=0)
        ax = plot_metric.plot_confusion_matrix(
            conf_mat, ax=axs[uniq_treatments.index(treatment)][uniq_types.index(time_type)],
            annot=True, fmt="d", title=f"{year_map_dict[time_type]}\nTreatment: {treatment}",
            cmap=sns.light_palette("seagreen", as_cmap=True), 
            linecolor='black', linewidths=0.5, cbar=False
        )
        selected[uniq_treatments.index(treatment)][uniq_types.index(time_type)] = 1

    remove_axs = axs[np.where(selected == 0)]
    for ax in remove_axs.ravel():
        ax.remove()

    if suptitle:
        fig.suptitle(suptitle)
    else:
        fig.suptitle(' '.join([experiment_prefix.capitalize(), dataset_name, estimator_type, 'Confusion Matrix', ' '.join(map(str.capitalize, experiment_suffix.split('_')))]))
    plt.tight_layout(pad=3)
    if save_path:
        plt.savefig(save_path)
    elif save_path is None:
        save_path = f'{experiment_prefix}_{estimator_type}_{dataset_name}_confmat_{experiment_suffix}.jpg'
        plt.savefig(experiment_folder / save_path)
    plt.show()

def plot_baseline_metric_by_treatment_type(
        project_label: pd.DataFrame, 
        running_baseline_conf_mat_dict: Dict[str, List[np.array]],
        suptitle: str=None, 
        save_path: str=None, 
        estimator_type: str=None,
        experiment_prefix: str='train',
        experiment_suffix: str=None,
        experiment_folder: Path=None,
        dataset_name: str='Dummy',
    ):
    # plot total accuracy for each of type-treatment pair
    types = project_label.columns.get_level_values(0)
    treatments = project_label.columns.get_level_values(1)
    uniq_types = list(dict.fromkeys(types))
    uniq_types = sorted(uniq_types, key=lambda x: treatment_time_order[x])
    uniq_treatments = list(dict.fromkeys(treatments))
    uniq_treatments = sorted(uniq_treatments, key=lambda x: treatment_type_order[x])

    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(12, 16))
    axs = axs.ravel()
    x = np.arange(len(uniq_types))
    width = 0.1

    strat_colors = plt.cm.get_cmap('Accent')
    strat_handles = []

    for dummy_i, dummy_strat in enumerate(running_baseline_conf_mat_dict.keys()):
        metric_dict = {}
        metric_names = ['accuracy', 'precision', 'recall', 'f_score']
        for treatment in uniq_treatments:
            metric_dict[treatment] = {}
            for metric in metric_names:
                metric_time_dict = OrderedDict()
                for time_horizon in uniq_types:
                    metric_time_dict[time_horizon] = [np.nan, np.nan]
                metric_dict[treatment][metric] = metric_time_dict

        for i, (time_type, treatment) in enumerate(project_label.columns):
            running_conf_mat = np.array(running_baseline_conf_mat_dict[dummy_strat])[:, i, :, :]
            metric_dict[treatment]['accuracy'][time_type] = ((running_conf_mat[:, 0, 0] + running_conf_mat[:, 1, 1]) / running_conf_mat.sum(axis=(1,2)))
            metric_dict[treatment]['precision'][time_type] = (running_conf_mat[:, 1, 1] / running_conf_mat[:, :, 1].sum(axis=1))
            metric_dict[treatment]['recall'][time_type] = (running_conf_mat[:, 1, 1] / running_conf_mat[:, 1, :].sum(axis=1))
            metric_dict[treatment]['f_score'][time_type] = (2 / (1 / metric_dict[treatment]['precision'][time_type] + 1 / metric_dict[treatment]['recall'][time_type]))

        treatment_handles = []
        for i, treatment in enumerate(uniq_treatments):
            violins = []
            violins.append(axs[0].violinplot(list(metric_dict[treatment]['accuracy'].values()), positions=x+i*width, widths=width, showmeans=False, showextrema=False))
            violins.append(axs[1].violinplot(list(metric_dict[treatment]['precision'].values()), positions=x+i*width, widths=width, showmeans=False, showextrema=False))
            violins.append(axs[2].violinplot(list(metric_dict[treatment]['recall'].values()), positions=x+i*width, widths=width, showmeans=False, showextrema=False))
            violins.append(axs[3].violinplot(list(metric_dict[treatment]['f_score'].values()), positions=x+i*width, widths=width, showmeans=False, showextrema=False))
            for violin_metrics in violins:
                for pc in violin_metrics["bodies"]:
                    pc.set_edgecolor(strat_colors(dummy_i))
                    pc.set_linewidth(3)
                    pc.set_alpha(1)
                    pc.set_facecolor(treatment_colors[treatment])
            treatment_handles.append(mpatches.Patch(color=treatment_colors[treatment]))
        strat_handles.append(mpatches.Patch(linewidth=4, edgecolor=strat_colors(dummy_i), facecolor="none"))

    for metric in range(len(metric_names)):
        axs[metric].set_ylabel(metric_names[metric])
        axs[metric].set_xticks(x + width * (len(uniq_treatments) - 1) / 2)
        axs[metric].set_xticklabels([year_map_dict[treatment_time] for treatment_time in uniq_types], rotation=10)
        axs[metric].legend(treatment_handles + strat_handles, uniq_treatments + list(running_baseline_conf_mat_dict.keys()), bbox_to_anchor=(1, 1), loc="upper left", title='Treatment Category and Naive Strategy')
        axs[metric].grid(True)
        axs[metric].set_ylim((-0.1, 1.1))

    if suptitle is not None:
        fig.suptitle(suptitle)
    else:
        fig.suptitle(' '.join([experiment_prefix.capitalize(), dataset_name, estimator_type, 'Baseline Results', ' '.join(map(str.capitalize, experiment_suffix.split('_')))]))
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    elif save_path is None:
        save_path = f'{experiment_prefix}_{estimator_type}_{dataset_name}_violin_baseline_{experiment_suffix}.jpg'
        plt.savefig(experiment_folder / save_path)
    
    plt.show()