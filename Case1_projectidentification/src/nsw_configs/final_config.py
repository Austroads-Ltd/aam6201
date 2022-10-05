import pandas as pd

from data import DATA_DIR # path to data directory
from src.util import ConfigNamespace

crack_data = DATA_DIR / "raw" / "NSW" / "crack_2015_2020.xlsx"
dtims_in_data = DATA_DIR / "raw" / "NSW" / "dTIMS_Extraction_3_yr_comm.xlsx"
dtims_out_data = DATA_DIR / "raw" / "NSW" / "dTIMS_output_10_year_forward_work_program.xlsx"
profile_data = DATA_DIR / "raw" / "NSW" / "prof_2015_to_2020.xlsx"
deflection_data = DATA_DIR / "raw" / "NSW" / "remaining_life_deflection.xlsx"
preprocessed_data = DATA_DIR / "processed" / "nsw" / "preprocessed_final.csv"

state_save_path = DATA_DIR.parent / "models" / "preprocessing_states" / "nsw"
eda_output = DATA_DIR.parent / 'reports' / 'eda' / 'nsw'

train_data = DATA_DIR / "processed" / "nsw" / "final" / "train_all.csv"
label_data = DATA_DIR / "processed" / "nsw" / "final" / "labels_all.csv"


def notna(x):
    return x.notna()

def notna_positive(x):
    return x.notna() & (x > 0)

def str_join(df):
    return df.astype(str).agg(''.join, axis=1)


CONFIG = ConfigNamespace({
    "name": "nsw-austroads",  # name of the configuration
    "target": None,  # target of the classification / regression task
    "problem_class": "classification",
    "data": {  # all keywords inside each step will be passed into the read function. Only source is required
        "eda": {
            "source": dtims_in_data,
        },
        "preprocessing": {
            "source": dtims_in_data,
        },
        "modelling": {
            "train_source": train_data,
            "label_source": label_data,
        }
    },
    "preprocessing": {  # TODO: Group these better
        "cat_encoding": {
            "OneHot": {
                "cat_col_list": ["Pavement Type", "Surface Material"],  # list of categorical features
                "encoding_count": [2, 1], # corresponding list of the maximum number of categories to encode the categorical features by, null defaults to the number of unique values
                "keep_other": [False, False] # set to True to keep the column Other (any value not falling into encoding count most popular values). list correspond to feature list
            },
            "Ordinal": {
                "cat_col_list": [], # list of categorical features
                "encoding_count": [None], # corresponding list of the maximum number of categories to encode the categorical features by, null defaults to the number of unique values
                "keep_other": [False] # set to True to keep the column Other (any value not falling into encoding count most popular values). list correspond to feature list
            }
        },
        "imputing": {
            "groupby_first": {
                "feature_list": [], # list of features to be imputed by groupings. Empty groups will be processed by settings below
                "groupby_subset": [], # list of features for grouping
                "sort_subset": [] # list of features for sorting the groups
            }, 
            "feature_list": [],
            "leave_out": False,  # set to true to impute all features but the one in feature_list, false to impute only the feature in feature_list
        }, 
        "filtering": { # for each column, define a function that accepts a pandas series and returns a boolean series, true if that value is to be kept.
            # Lambda functions can be passed in but can't be pickled and the config cannot be saved
            'ELEMENTID': notna,
            'Crack%': notna,
            "iri_owp": notna,
            "iri_iwp": notna,
            "Rutting mm": notna,
            "rut_owp": notna,
            'rut_iwp': notna,
            'AADT': notna,
            "HeavyIndex": notna,
            "DI_TXT": notna,
            "txt_mid": notna,
            "Pavement Type": notna,
            "Surface Material": notna,
            "D0": notna,
            "D200": notna,
        },
        "feature_removal": {
            # list of features
            'feature_list': [
                "ELEMENTID", "Crack%", "iri_owp", "iri_iwp",
                "Rutting mm", "rut_owp", "rut_iwp", "AADT", "HeavyIndex", 
                "DI_TXT", "txt_mid", "Pavement Type", "Surface Material",
                "Pavement age", "Surface age", "D0", "D200"
            ],
            'drop': False # set to true to drop only features in feature_list, false to drop ALL features not in feature_list
        },
        "normalizing": {
            'feature_list': [
                "Crack%", "iri_owp", "iri_iwp", 
                "Pavement age", "Surface age",
                "Rutting mm", "rut_owp", "rut_iwp", 
                "AADT", "HeavyIndex",
                "DI_TXT", "txt_mid", 
                "D0", "D200"
            ],
            'leave_out': False, # set to true to normalise all features but those in feature list, false to normalise only features in feature_list
        },
        "state_save_path": state_save_path, # path to dump states of preprocessing objects
        "save_complete": {
            "flag": True,
            "save_path": preprocessed_data,
            "save_method": "save_csv" # must match option src.util.save_complete_data
        }
    },
    "sampling": {
        "n_sample_per_fold": 1, # number of random samplings of the original dataset
        "kfold": 5,
        "sample_size": 0.7, # size of the sample, an integer for a strict number of data points, or a float as a fraction of the datset
        "method": None,
        "test_size": 0.3, # length(test) / length(all), as in sklearn.model_selection.train_test_split
        "method_params": { # dictionary of params for a method
            "index_row": str_join, # a method which outputs a hashable value for each row of a dataframe. Takes the whole dataframe as input and outputs a dataseries with the same index
            "on_label": True # set to true to apply index_row on the set of labels, false to apply on the set of train samples
        }
    },
    "training": {
        "show_pdp": False, # not supported yet
    },
    "visualisation": { # configuration for visualisation scripts
        "eda": {
            "save_path": eda_output
        }
    }
})

class FeatureAdder():
    """Feature Adder which adds new features this config will require"""
    def __call__(self, all_df: pd.DataFrame):

        # -- rename
        all_df = all_df.rename(columns={
            'DI_TYPE_PAVE': 'Pavement Type',
            'DI_TYPE_SURF': 'Surface Material',
            'DI_TRF_AADT': 'AADT',
            'DI_CRK_ALL_AREA': 'Crack%',
            "DI_D0": 'D0',
            "DI_D200": 'D200',
            "DI_RUT": "Rutting mm",
            "DI_PCT_HV": "HeavyIndex",
            "age": "Pavement age",
            "age_surface": "Surface age",
        })

        # -- rename values
        all_df.loc[:, 'Pavement Type'] = all_df['Pavement Type'].replace({
            'F': 'Flexible',
            'R': 'Rigid'
        })

        return all_df
