"""
Define constant config
"""

from data import DATA_DIR # path to data directory
from src.util import ConfigNamespace

if (DATA_DIR / "processed" / "VIC" / "final").exists() is False:
    (DATA_DIR / "processed" / "VIC" / "final").mkdir(parents=True)

raw_data = DATA_DIR / "raw" / "vic" / "AAM6201 Data Reques" / "PCS data" / "PCS 2006-07 09 Aug 21.xlsx"
pcs_data_dir = DATA_DIR / "raw" / "vic" / "AAM6201 Data Reques" / "PCS data"
complete_data = DATA_DIR / "processed" / "VIC" / "final" / "vic_pcs_2006.csv"
state_save_path = DATA_DIR.parent / "models" / "preprocessing_states" / "vic" # root directory is parent of data directory
eda_output = DATA_DIR.parent / 'reports' / 'eda' / 'vic'

train_data = DATA_DIR / "processed" / "VIC" / "final" / "train_all.csv"
label_data = DATA_DIR / "processed" / "VIC" / "final" / "labels_all.csv"


def notna(x):
    return x.notna()


def notna_positive(x):
    return x.notna() & (x > 0)


def str_join(df):
    return df.astype(str).agg(''.join, axis=1)


CONFIG = ConfigNamespace({
    "name": "vic-austroads",  # name of the configuration
    "target": None,  # target of the classification / regression task
    "problem_class": "classification",
    "data": {  # all keywords inside each step will be passed into the read function. Only source is required
        "eda": {
            "source": raw_data,
        },
        "preprocessing": {
            "source": pcs_data_dir,
            "exclude": ["PCS2018-19.xlsx"]
        },
        "modelling": {
            "train_source": train_data,
            "label_source": label_data,
        }
    },
    "preprocessing": {  # TODO: Group these better
        "cat_encoding": {
            "OneHot": {
                "cat_col_list": ["Road Maintenance Category", "Pavement Maintenance Category"],  # list of categorical features
                "encoding_count": [4, 3], # corresponding list of the maximum number of categories to encode the categorical features by, null defaults to the number of unique values
                "keep_other": [False, False] # set to True to keep the column Other (any value not falling into encoding count most popular values). list correspond to feature list
            },
            "Ordinal": {
                "cat_col_list": ["Is_Distressed"], # list of categorical features
                "encoding_count": [None], # corresponding list of the maximum number of categories to encode the categorical features by, null defaults to the number of unique values
                "keep_other": [False] # set to True to keep the column Other (any value not falling into encoding count most popular values). list correspond to feature list
            }
        },
        "imputing": {
            "groupby_first": {
                "feature_list": ["Avg_Rgh_Iri", "Avg_Rutting", "HATI", "Is_Distressed", "Texture_C", "Texture_L", "Road Maintenance Category", "Pavement Maintenance Category", "Texture_Loss", "Rutting_Sd_Lane", "Rutting_Sd_Lwp", "Rutting_Sd_Rwp"], # list of features to be imputed by groupings. Empty groups will be processed by settings below
                "groupby_subset": ["Road_Number", "Direction", "Survey Date"], # list of features for grouping
                "sort_subset": ["From_Measure"] # list of features for sorting the groups
            }, 
            "feature_list": [],
            "leave_out": False,  # set to true to impute all features but the one in feature_list, false to impute only the feature in feature_list
        }, 
        "filtering": { # for each column, define a function that accepts a pandas series and returns a boolean series, true if that value is to be kept.
            # Lambda functions can be passed in but can't be pickled and the config cannot be saved
            'Road_Number': notna,
            'Direction': notna,
            'From_Measure': notna,
            'Survey Date': notna,
            'Length_KM': notna_positive,
            "Avg_Rgh_Iri": notna_positive,
            "Avg_Rutting": notna_positive,
            "HATI": notna_positive,
            "Texture_C": notna_positive,
            "Texture_L": notna_positive,
            "Texture_Loss": notna_positive,
            "Rutting_Sd_Lane": notna_positive,
            "Rutting_Sd_Lwp": notna_positive,
            "Rutting_Sd_Rwp": notna_positive
        },
        "feature_removal": {
            'feature_list': ["Road_Number", "Direction", "Survey Date", "From_Measure", "Avg_Rgh_Iri", "Avg_Rutting", "HATI", "Is_Distressed", "Texture_C", "Texture_L", "Road Maintenance Category", "Pavement Maintenance Category", 'Length_KM', "Texture_Loss", "Rutting_Sd_Lane", "Rutting_Sd_Lwp", "Rutting_Sd_Rwp"], # list of features
            'drop': False # set to true to drop only features in feature_list, false to drop ALL features not in feature_list
        },
        "normalizing": {
            'feature_list': ["Avg_Rutting", "Avg_Rgh_Iri", "Texture_C", "Texture_L",  "Texture_Loss", "Rutting_Sd_Lane", "Rutting_Sd_Lwp", "Rutting_Sd_Rwp"],
            'leave_out': False, # set to true to normalise all features but those in feature list, false to normalise only features in feature_list
        },
        "state_save_path": state_save_path, # path to dump states of preprocessing objects
        "save_complete": {
            "flag": True,
            "save_path": complete_data,
            "save_method": "save_csv" # must match option src.util.save_complete_data
        }
    },
    "sampling": {
        "n_sample": 20, # number of random samplings of the original dataset
        "sample_size": 0.8, # size of the sample, an integer for a strict number of data points, or a float as a fraction of the datset
        "method": "balanced",
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
