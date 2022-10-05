import pandas as pd
import numpy as np

from data import DATA_DIR # path to data directory
from src.util import ConfigNamespace
from functools import partial

raw_data = DATA_DIR / "raw" / "MRWA" / "AAM6201_MRWA" / "HSD_TSD Condition Data.csv"
complete_data = DATA_DIR / "processed" / "HSD_TSD_processed.csv"
state_save_path = DATA_DIR.parent / "models" / "preprocessing_states" / "mrwa" # root directory is parent of data directory
eda_output = DATA_DIR.parent / 'reports' / 'eda' / 'mrwa'
figure_output = DATA_DIR.parent / 'reports' / 'figure' / 'mrwa'

processed_train = DATA_DIR / "processed" / "MRWA" / "train_processed_nzta_mrwa_transfer.csv"
processed_valid = DATA_DIR / "processed" / "MRWA" / "valid_flattened_nzta_mrwa_transfer.csv"

def as_str(x: pd.DataFrame):
    return x.astype(str).agg(''.join, axis=1)

def not_neg(x: pd.Series):
    return x >= 0

def not_neg_nan(x: pd.Series):
    return (x >= 0) | (np.isnan(x))

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
    "eda": {
        "source": raw_data,
    },
    "preprocessing": {
        "source": raw_data,
    },
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
            "groupby_subset": ['RoadID', 'Date of condition data', 'Direction'], # list of features for grouping 
            "sort_subset": ['Start'] # list of features for sorting the groups
        }, 
        "feature_list": ['RoadID', 'Date of condition data', 'Start', 'Direction', 'End'],
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
        'Crack%': not_neg_nan,
    },
    "feature_removal": {
        'feature_list': [
            'RoadID', 'Date of condition data', 'Start', 'Direction', 'End', 
            'Pavement Type', 'Surface Material', 'Surface age', 'Pavement age',
            'AADT', 'HeavyIndex', 'IRI', 'Rutting mm',
            'Crack%', 'D0', 'D200'
        ], # list of features
        'drop': False # set to true to drop only features in feature_list, false to drop ALL features not in feature_list
    },
    "normalizing": {
        'feature_list': ['RoadID', 'Date of condition data', 'Start', 'Direction', 'End',
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
            "save_path": processed_train,
            "save_method": "save_csv" # must match option src.util.save_complete_data
        },
        "valid": {
            "flag": True,
            "save_path": processed_valid,
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
    """Feature Adder which adds new features this config will require"""
    def __call__(self, all_df):
        all_df = all_df.copy()
        #--
        all_df['Surface Material'] = all_df['Asphalt/Seal Type'].replace({
            'Single Seal': 'SS',
            'DD Seal': 'SS',
        }, regex=True)

        #--
        all_df.loc[:, 'Pavement Type'] = all_df['Pavement Type'].fillna('Other')

        #--
        all_df = all_df.rename(columns={
            'Cway Roughness IRI': 'IRI',
            'Cway Rut depth mm': 'Rutting mm',
            'Cway Cracking %': 'Crack%',
            'Perc_Heavy': 'HeavyIndex'
        })

        # generate surface age
        all_df.loc[:, 'Surface age'] = all_df['Date of condition data'].astype(np.datetime64).dt.year - all_df['Surface year']
        all_df.loc[:, 'Pavement age'] = all_df['Date of condition data'].astype(np.datetime64).dt.year - all_df['Pavement age']

        return all_df
