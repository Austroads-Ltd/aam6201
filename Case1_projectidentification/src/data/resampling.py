"""
Provide the following functionalities:
- resampling from a dataset with the bootstrap strategy
- performing train-test split on the sampled dataset
"""
import pandas as pd
import numpy as np

from tqdm.auto import tqdm
from math import ceil, floor
from sklearn.model_selection import train_test_split
from typing import Tuple, Generator

def get_stratify_target(df: pd.DataFrame, config: dict) -> pd.Series:
    """
    Return labels for stratified split given dataframe
    """
    assert config['problem_class'] == 'classification', 'Balanced resampling is applicable only to a classification problem!'
    try:
        stratify: pd.Series = config['sampling']['method_params']['index_row'](df)
    except TypeError:
        raise TypeError('Method for indexing each row is not defined for balanced resampling.')
    
    return stratify

def resample_with_split(feature_data: pd.DataFrame, label_data: pd.DataFrame, config: dict) ->  Generator[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame], None, None]:
    """
    Given a dataframe for features and labels and settings from the configuration file, yield a list of dataframes, each of which is a subsample of the original under a certain strategy. These can then be used for training and validation.\n

    Args:
        feature_data: DataFrame of data points. Currently only support dataframes
        label_data: DataFrame of labels for `feature_data`
        config: the configuration of the experiment

    Return:
    \tx_train, x_test, y_train, y_test
    """
    method = config['sampling']['method']
    n = config['sampling']['n_sample_per_fold']
    test_size = config['sampling']['test_size']
    sample_size = config['sampling']['sample_size']
    rand_gen = np.random.RandomState(config['random_seed']) # fixed a random sequence of numbers by seed


    # calculate target if necessary
    if method == 'balanced':
        target_df = label_data if config['sampling']['method_params']['on_label'] else feature_data
        stratify = get_stratify_target(target_df, config)
    else:
        stratify = None

    # calculate sizes
    int_test_size = ceil(test_size * len(feature_data)) # how sklearn does
    int_train_size = len(feature_data) - int_test_size
    int_sample_size = sample_size if sample_size > 1 and isinstance(sample_size, int) else ceil(sample_size * int_train_size)    
    if test_size == 0:
        print("Warning: test size is set to 0. There will be NO train-test split.")

    # train-test split
    for _ in tqdm(range(config['sampling']['kfold']), desc='Fold'):
        if test_size > 0:
            train_feature, test_feature, train_label, test_label = train_test_split(
                feature_data, 
                label_data, 
                test_size=test_size, 
                stratify=stratify,
                random_state=rand_gen.random_integers(1, 1000)
            )
        else:
            train_feature, test_feature, train_label, test_label = feature_data, feature_data.sample(0), label_data, label_data.sample(0)

        # safety check
        if test_size > 0:
            assert set(train_feature.index).intersection(set(test_feature.index)) == set() 
            assert set(train_label.index).intersection(set(test_label.index)) == set() 
            if stratify is not None:
                assert set(stratify.loc[train_feature.index]) == set(stratify.loc[test_feature.index]), 'Test set does not have the same classes as train set' # train test has the same classes

        for _ in tqdm(range(n), desc='Sample', leave=False):
            train_sample, train_sample_label = random_sample_from(
                train_feature, 
                train_label, 
                rand_gen, 
                method=method, 
                stratify=stratify.loc[train_feature.index] if stratify is not None else None, 
                size=int_sample_size
            )
            yield train_sample, test_feature, train_sample_label, test_label 


def random_sample_from(df: pd.DataFrame, labels: pd.DataFrame, rand_gen: np.random.RandomState, method: str='random', stratify: pd.Series=None, size: int=100):
    """Draw a random sample from inputted df under a given strategy"""
    
    if (method is None) or (method == "none"):
        # perform no sampling
        return df, labels
        
    if method == 'random':
        # randomly choose size out of df
        iloc_idx = rand_gen.choice(range(len(df)), size=size, replace=True)
        train_x, train_y = df.iloc[iloc_idx], labels.iloc[iloc_idx]
        return train_x, train_y 

    elif method == 'balanced': # only work with classification problem 
        train_per_class_count = floor(size / stratify.nunique())
        train_iloc_idx = np.concatenate([
            rand_gen.choice(stratify[stratify == train_target_cls].index, size=train_per_class_count, replace=True) \
            for train_target_cls in stratify.unique()
        ], axis=0)
        # fill up to int_sample_size
        train_iloc_idx = np.append(train_iloc_idx, rand_gen.choice(df.index, replace=True, size=size-len(train_iloc_idx))) 

        train_x, train_y = df.loc[train_iloc_idx], labels.loc[train_iloc_idx]
        return train_x, train_y 
    else:
        raise NotImplementedError(f"Given method {method} not implemented")