"""
Perform routine EDA tasks:
- show description and information on all features
- show histograms of non-numerical features' top 10 most popular values
- show distributions of numerical features
- show number of unique values for non-numerical features
- show cross-validation

Contains code written by: 
- Catherine Yu (Data Scientist / Engineer)
- David Rawlinson (Lead Data Scientist / Engineer)
- Rafid Morshedi (Senior Data Scientist / Engineer)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.visualization.plot_metric import plot_cross_correlation
from src.util import load_data
from pathlib import Path
from math import ceil, sqrt

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


def show_info_and_description(df: pd.DataFrame, save_path: Path=None) -> None:
    """
    Use pandas descriptive methods, save to text file optionally
    """
    num_desc = df.describe() if df.select_dtypes(include=['float64', 'int64']).shape[1] > 0 else None
    cat_desc = df.describe(include=object) if df.select_dtypes(include=['object']).shape[1] > 0 else None

    f = open(save_path, 'a') if save_path else None

    print('Standard information of dataframe: ', file=f)
    df.info(buf=f)
    if num_desc is not None:
        print('\nInformation on numerical features: ', file=f)
        print(num_desc, file=f)
    else:
        print('\n No numerical columns exist.', file=f)
    if cat_desc is not None:
        print('\nInformation on categorical features: ', file=f)
        print(cat_desc, file=f)
    else:
        print('\n No non numerical column exists.', file=f)

    if f is not None:
        f.close()

def show_histogram_numeric_features(df: pd.DataFrame, save_path: Path=None) -> None:
    """
    Plot distributions of all numeric features of a given dataframe.
    Optionally save a figure
    """
    numeric_df = df.select_dtypes(include=['float64','int64'])
    
    if numeric_df.shape[1] < 0:
        print("No numerical columns exist")
        return

    edge_length = ceil(sqrt(numeric_df.shape[1])) # edge_length ** 2 >= number of plots, thus filling a square
    numeric_df.hist(figsize=(2 * edge_length, 2 * edge_length), bins=50, xlabelsize=8, ylabelsize=8)
    plt.tight_layout()

    if save_path:
        plt.savefig(str(save_path.resolve()))
    else:
        plt.show()

def show_count_per_unique_value_non_numeric_features(df: pd.DataFrame, save_path: Path=None) -> None:
    """
    Show counts of samples with unique values for each non-numeric feature in a dataframe
    """
    all_df_obj = df.select_dtypes(include=['object'])
    f = open(save_path, 'a') if save_path else None
    print("\n Number of samples per unique value for non numerical features:\n", file=f)
    # table of number of samples per unique values of non numerical features
    if all_df_obj.shape[1] > 0:
        stack = []
        for col in all_df_obj.columns:
            value_counts = pd.DataFrame(all_df_obj[col].value_counts().sort_values())
            value_counts = pd.concat([value_counts], keys=[col], names=['Feature', 'Values']).rename(columns={col: 'Count'})
            stack.append(value_counts)
        
        print(pd.concat(stack, axis=0).transpose(), file=f)
    else:
        print("No categorical column exist to show counts of samples with unique values", file=f)

    if f is not None:
        f.close()

def show_frequency_non_numeric_features(df: pd.DataFrame, save_path: Path=None, show_date: bool=False) -> None:
    """
    Plot bar plot showing the frequency of unique values of non numeric features

    Args:
        show_date: if True, include datetime features in plot
    """
    nonnum_df = df.select_dtypes(include=['object'])
    if show_date is False:
        nonnum_df = nonnum_df.select_dtypes(exclude=['datetime'])
    
    if nonnum_df.shape[1] == 0:
        print("No non numerical columns exist")
        return
    
    edge_length = ceil(sqrt(nonnum_df.shape[1])) # edge_length ** 2 >= number of plots, thus filling a square
    fig, axs = plt.subplots(edge_length, edge_length, figsize=(edge_length * 2, edge_length * 2))
    axs = np.ravel(axs)

    # plot top 10 values of each
    for i, col in enumerate(nonnum_df.columns):
        nonnum_df[col].value_counts().iloc[:10].plot(kind='bar', subplots=True, ax=axs[i])
        axs[i].set_title(col)
    
    fig.tight_layout()

    if save_path:
        plt.savefig(str(save_path.resolve()))
    else:
        plt.show()

def show_boxplot_numeric_features(df: pd.DataFrame, save_path: Path=None) -> pd.Series:
    """
    Show boxplots of numerical features in a given dataframe, optionally save a figure.
    """
    numeric_df = df.select_dtypes(include=['float64','int64'])
    
    if numeric_df.shape[1] < 0:
        print("No numerical columns exist")
        return

    edge_length = ceil(sqrt(numeric_df.shape[1])) # edge_length ** 2 >= number of plots, thus filling a square
    fig, axs = plt.subplots(edge_length, edge_length, figsize=(3 * edge_length, 3 * edge_length)) 
    axs = np.ravel(axs)
    for i, col in enumerate(numeric_df.columns):
        numeric_df[col].plot(kind='box', ax=axs[i])
    fig.tight_layout()

    if save_path:
        plt.savefig(str(save_path.resolve()))
    else:
        plt.show()

def show_count_unique_values_non_numeric_features(df: pd.DataFrame, save_path: Path=None) -> pd.Series:
    """
    Show count of unique values in non numeric features, optionally print to a file.
    Return pandas series indexed by non numeric features with count of unique values
    """
    nonnum_df = df.select_dtypes(include=['object'])
    f = open(save_path, 'a') if save_path else None
    print("\n Number of unique values for non numerical features:\n", file=f)
    for col in nonnum_df.columns:
        print(col + ' unique items = ' + str(len(set(nonnum_df[col].tolist()))), file=f)
    if f is not None:
        f.close()
    return nonnum_df.nunique()

def show_cross_correlation(df: pd.DataFrame, save_path: Path=None) -> pd.DataFrame:
    """
    Plot cross correlation between numerical columns of a dataframe
    Return the correlation matrix
    """
    corr = df.copy()
    corr_matrix = corr.corr().abs()
    plot_cross_correlation(corr_matrix, save_path=save_path)
    return corr_matrix
