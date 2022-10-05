"""
Define constant config
"""
import pandas as pd
import numpy as np

from data import DATA_DIR # path to data directory
from functools import partial
from src.util import ConfigNamespace

state_save_path = DATA_DIR.parent / 'models' / 'preprocessing_states' / 'nzta'

def as_str(x: pd.DataFrame):
    return x.astype(str).agg(''.join, axis=1)

def not_neg(x: pd.Series):
    return x >= 0

def within(x: pd.Series, a, b):
    return (x > a) & (x < b)

CONFIG = ConfigNamespace({
    "name": "mrwa-austroads-nzta-transfer", # name of the configuration
    "target": None, # target of the classification / regression task
    "problem_class": "classification",
    "random_seed": 10,
})

DATA_CONFIG = ConfigNamespace({
    # all keywords inside each step will be passed into the read function. Only source is required
})

PREPROCESSING_CONFIG = ConfigNamespace({
    "cat_encoding": {
        "OneHot": {
            "cat_col_list": ['Pavement Type', 'Surface Material'], # list of categorical features
            "encoding_count": [2, 1], # corresponding list of the maximum number of categories to encode the categorical features by, null defaults to the number of unique values
            "keep_other": [False, False] # set to True to keep the column Other (any value not falling into encoding count most popular values). list correspond to feature list
        },
        "Ordinal": {
            "cat_col_list": [], # list of categorical features
            "encoding_count": [], # corresponding list of the maximum number of categories to encode the categorical features by, null defaults to the number of unique values
            "keep_other": [] # set to True to keep the column Other (any value not falling into encoding count most popular values). list correspond to feature list
        }
    },
    "imputing": {
        "groupby_first": {
            "feature_list": [
                'Pavement Type', 'Surface Material', 
                'Surface age', 'Pavement age', 'AADT', 'HeavyIndex', 
                'IRI', 'Rutting mm',
                'D0', 'D200', # higher deflection are too highly correlated with each other
                'Crack%', # high missing
            ], # list of features to be imputed by groupings. Empty groups will be processed by settings below
            "groupby_subset": ['RoadID', 'Date of condition data'], # list of features for grouping 
            "sort_subset": ['Start'] # list of features for sorting the groups
        }, 
        "feature_list": ['RoadID', 'Date of condition data', 'Start', 'End'],
        "leave_out": True, # set to true to impute all features but the one in feature_list, false to impute only the feature in feature_list
    },
    "filtering": { # for each column, define a function that accepts a pandas series and returns a boolean series, true if that value is to be kept.
        # Lambda functions can be passed in but can't be pickled and the config cannot be saved
        'Surface age': partial(within, a=0, b=80),
        'Pavement age': partial(within, a=0, b=80),
        'AADT': not_neg,
        'HeavyIndex': partial(within, a=0, b=100),
        'IRI': not_neg,
        'Rutting mm': not_neg,
        'Crack%': not_neg,
    },
    "feature_removal": {
        'feature_list': [
            'RoadID', 'Date of condition data', 'Start', 'End', 
            'Pavement Type', 'Surface Material', 'Surface age', 'Pavement age',
            'AADT', 'IRI', 'Rutting mm', 'HeavyIndex',
            'Crack%', 'D0', 'D200',
        ], # list of features
        'drop': False # set to true to drop only features in feature_list, false to drop ALL features not in feature_list
    },
    "normalizing": {
        'feature_list': ['RoadID', 'Date of condition data', 'Start', 'End',
            'Pavement Type_Flexible', 
            'Pavement Type_Rigid',
            'Surface Material_SS',
        ],
        'leave_out': True, # set to true to normalise all features but those in feature list, false to normalise only features in feature_list
    },
    "state_save_path": state_save_path, # path to dump states of preprocessing objects
    "save_complete": {
        'train': {
            "flag": True,
            "save_method": "save_csv" # must match option src.util.save_complete_data
        },
        "valid": {
            "flag": True,
            "save_method": "save_csv"
        }
    }
})

SAMPLING_CONFIG = ConfigNamespace({
    "n_sample_per_fold": 1, # number of random samplings of the original dataset
    "kfold": 5,
    "sample_size": 1, # size of the sample, an integer for a strict number of data points, or a float as a fraction of the datset
    "method": None,
    "test_size": 0.3, # length(test) / length(all), as in sklearn.model_selection.train_test_split
    "method_params": { # dictionary of params for a method
        "index_row": as_str, # a method which outputs a hashable value for each row of a dataframe or dataseries. Takes the whole dataframe or series as input and outputs a dataseries with the same index
        "on_label": True # set to true to apply index_row on the set of labels, false to apply on the set of train samples
    }
})

CONFIG['data'] = DATA_CONFIG
CONFIG['preprocessing'] = PREPROCESSING_CONFIG
CONFIG['sampling'] = SAMPLING_CONFIG

class FeatureAdder():

    def __call__(self, all_df: pd.DataFrame):
        all_df = all_df.copy()

        # -- Rutting
        all_df.loc[:, 'IRI'] = (all_df['NAASRA'] + 1.27) / 26.49
        all_df = all_df.rename(columns={'Rutting Mean': 'Rutting mm'})

        # -- surface structure
        all_df.loc[:, 'Surface Material'] = all_df['Surface Material'].replace({
            'Single Coat Seal': 'SS',
            'Two Coat Seal': 'SS',
        }, regex=True)
        all_df.loc[:, 'Surface age'] = (all_df['Date of condition data'] - all_df['Surfacing Date']) / np.timedelta64(1, 'Y')

        # -- traffic
        all_df = all_df.rename(columns={
            'ADT': 'AADT',
            '% Heavy Vehicles': 'HeavyIndex'
        })

        # -- pavement
        # -- type
        all_df.loc[:, 'Pavement Type'] = all_df['Pavement Type'].replace({
            '.*Flexible.*': 'Flexible',
            '.*Concrete.*': 'Rigid',
            '(Bridge|Unsealed)': 'Other' 
        }, regex=True)
        all_df.loc[:, 'Pavement Type'] = all_df['Pavement Type'].fillna('Other')
        assert set(all_df['Pavement Type']) == {'Flexible', 'Rigid', 'Other'}

        # -- cracking + deflection
        all_df = all_df.rename(columns={
            'crk_alligator': 'Crack%',
            'def_fwd': 'D0',
            'Curvature': 'D200'
        })

        # -- age
        all_df.loc[:, 'Pavement age'] = all_df['Date of condition data'].dt.year - (2020 - all_df['Pavement age'])

        # -- end
        all_df.loc[:, 'End'] = all_df['Start'] + 100

        return all_df