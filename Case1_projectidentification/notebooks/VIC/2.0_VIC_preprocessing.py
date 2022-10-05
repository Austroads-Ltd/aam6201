import os
os.getcwd()
os.chdir("../..")

import os
import numpy as np
import pandas as pd
from src.vic_configs.final_config import DATA_DIR
from tqdm import tqdm
import src.util as util
import src.features.preprocessing as preprocessing 

from src.vic_configs.final_config import CONFIG
from IPython.display import display

# load data
dfs = []
for fname in os.listdir(CONFIG['data']['preprocessing']['source']):
    if fname in CONFIG['data']['preprocessing']['exclude']:
        continue
    dfs.append(util.load_data(CONFIG['data']['preprocessing']['source']/fname))
all_df = pd.concat(dfs, ignore_index=True)
all_df = all_df.reset_index(drop=True)

# define classes
class FeatureEncodeImputeNormalizeContainer:
    """
    Container for encoding, imputation, and normalization operations, in that order.
    """

    def __init__(self):
        self.feature_encoding = None # remember feature encoding for future
        self.date_encoding = None # remember date encoding
        self.feature_scaling = None # remember feature scaling for future
        self.imputer_dict = {} # dictionary between columns and its imputer

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        # Perform imputation.
        if CONFIG['preprocessing']['imputing']['groupby_first']['feature_list']:
            df = preprocessing.groupby_impute(df, CONFIG)

        if len(self.imputer_dict) == 0:
            self.imputer_dict = preprocessing.fit_imputer(df, CONFIG)
        imputed_df = preprocessing.impute(self.imputer_dict, df)

        # encoding must be done after imputation, otherwise NA value is treated as a unique category unintentionally
        # Perform categorical encoding on specified variables
        if self.feature_encoding is None:
            self.feature_encoding = preprocessing.get_categorical_encoding(imputed_df , CONFIG)
        encoded_df = preprocessing.encode_categorical_features(imputed_df, CONFIG, self.feature_encoding)

        # Perform scaling
        if self.feature_scaling is None:
            try:
                self.feature_scaling = preprocessing.fit_scaler(encoded_df, CONFIG) # TODO: now we need to remember the scaler hasn't been fitted on CONFIG['target']. Is this good?
            except KeyError:
                raise KeyError(f"Target column {CONFIG['target']} is not in the dataframe's columns!")
        encoded_df = preprocessing.scale(encoded_df, self.feature_scaling, CONFIG)

        return encoded_df

# perform filtering on samples by thresholding against features 
class SampleFilterByFeatureThresholdContainer:
    """
    Container for filtering operations on the datset to remove unwanted rows. The index is not changed, however.
    """
    def __call__(self, df: pd.DataFrame):
        for col, key_fn in CONFIG['preprocessing']['filtering'].items():
            df = df[key_fn(df[col])] # remove height = 0 as they are invalid
        return df

# drop features
class FeatureRemovalContainer: 
    """
    Container for feature removal operations to remove unwanted features.
    """
    def __call__(self, df: pd.DataFrame):
        # remove by setting in config
        col_names = CONFIG['preprocessing']['feature_removal']['feature_list']
        drop = CONFIG['preprocessing']['feature_removal']['drop']
        if drop:
            df = df.drop(columns=col_names)
        else:
            df = df[col_names].copy()
        return df

# Initialise class containers
feature_preprocess = FeatureEncodeImputeNormalizeContainer()
sample_fitler = SampleFilterByFeatureThresholdContainer()
feature_removal = FeatureRemovalContainer()

# Sequential processing.
filtered_df = sample_fitler(all_df)
col_filtered_df = feature_removal(filtered_df)
complete_df = feature_preprocess(col_filtered_df)

# Saving completed dataset
util.save_complete_data(complete_df, **CONFIG['preprocessing']['save_complete'])

# Saving preprocessing states for use on validation datasets
state_dict = {
    'config': CONFIG,
    'feature_encoder': feature_preprocess.feature_encoding,
    'scaler': feature_preprocess.feature_scaling,
    'imputer_dict': feature_preprocess.imputer_dict
}
util.pickle_data(state_dict, CONFIG['preprocessing']['state_save_path'], 'preprocessing_state_dict.sav')

# clean projects
projects = util.load_data(DATA_DIR / 'raw' / 'VIC' / 'AAM6201 Data Reques' / 'Work Program' / 'Pavement Diary since 2014_2019.xlsx', sheet_name=0)
treatment_lookup = util.load_data(DATA_DIR.parent / "references" / "TreatmentCategory.csv")
treatment_lookup = treatment_lookup[treatment_lookup['Jurisdiction'] == 'VIC']

projects = projects.rename(columns={"Route Number": "Road_Number",
                                    "From Measure": "From_Measure"})

old_shape = projects.shape
cleaned_projects = projects.dropna(
    subset=['Road_Number', 'Direction', 'From_Measure', 'Length', 'Treatment Date', 'Treatment Type']
).copy()

cleaned_projects.loc[cleaned_projects['Direction'].str.contains('Forward'), 'Direction'] = 'Forward'
cleaned_projects.loc[cleaned_projects['Direction'].str.contains('Reverse'), 'Direction'] = 'Reverse'
old_shape = cleaned_projects.shape
cleaned_projects = cleaned_projects[cleaned_projects['Direction'].isin({'Forward', 'Reverse'})]

cleaned_projects["Treatment Category"] = cleaned_projects["Treatment Type"]
cleaned_projects["Treatment Category"] = cleaned_projects["Treatment Category"].replace(dict(zip(treatment_lookup["Specific Category Value"], treatment_lookup["Generic Category"])))
old_shape = cleaned_projects.shape
cleaned_projects = cleaned_projects.drop(index=cleaned_projects[~cleaned_projects["Treatment Category"].isin(treatment_lookup["Generic Category"])].index)

cleaned_projects = cleaned_projects[["Road_Number", "Route Name", "Direction", "From_Measure", "To Measure", "Length", "Treatment Date", "Treatment Category"]]
cleaned_projects = cleaned_projects.rename(columns={"Road_Number": "RoadID",
                                                    "From_Measure": "Start",
                                                    "Treatment Date": "Date Treatment"})
old_shape = cleaned_projects.shape
cleaned_projects = cleaned_projects.drop_duplicates()

old_shape = cleaned_projects.shape
cleaned_projects = cleaned_projects[cleaned_projects['Treatment Category'].notna()]

complete_df["Length"] = complete_df["Length_KM"] * 1000
complete_df = complete_df.rename(columns={"Road_Number": "RoadID",
                                          "From_Measure": "Start",
                                          "Survey Date": "Date of condition data"})
complete_df = complete_df.drop(columns=["Length_KM"])
cleaned_df = complete_df
cleaned_df.loc[:, 'Date of condition data'] = cleaned_df['Date of condition data'].astype(np.datetime64)
cleaned_projects.loc[:, 'Date Treatment'] = cleaned_projects['Date Treatment'].astype(np.datetime64)

def make_label_mat(grouped_labels: pd.DataFrame, treatments: list, latest_condition_date) -> pd.DataFrame:
    label_mat = pd.DataFrame(columns=treatments, index=[
        'Treatment within 1 year',
        'Treatment between 1 to 3 years',
        'Treatment between 3 to 5 years',
        'Treatment between 5 to 10 years',
        'Treatment between 10 to 30 years'
    ])
    label_mat.loc[:, :] = 0
    if len(grouped_labels) == 0:
        return label_mat

    for i, treatment in enumerate(treatments):
        category_labels = grouped_labels[grouped_labels['Treatment Category'] == treatment]
        if len(category_labels) == 0:
            continue
        
        year_offset = ((category_labels['Date Treatment'] - latest_condition_date) / np.timedelta64(1, 'Y'))

        if (year_offset <= 1).any():
            label_mat.iloc[0, i] = 1
    
        if ((year_offset > 1) & (year_offset <= 3)).any():
            label_mat.iloc[1, i] = 1

        if ((year_offset > 3) & (year_offset <= 5)).any():
            label_mat.iloc[2, i] = 1

        if ((year_offset > 5) & (year_offset <= 10)).any():
            label_mat.iloc[3, i] = 1
        
        if ((year_offset > 10) & (year_offset <= 30)).any():
            label_mat.iloc[4, i] = 1
    
    return label_mat

indexed_df = cleaned_df.drop_duplicates(['RoadID', 'Direction', 'Start'])

# define global variables
constant_value_columns = indexed_df.columns[indexed_df.columns.str.startswith("Road Maintenance Category") | indexed_df.columns.str.startswith("Pavement Maintenance Category")]
value_columns = indexed_df.drop(columns=['RoadID', 'Direction', 'Start', 'Length', 'Date of condition data']).columns
value_columns = set(value_columns) - set(constant_value_columns)
treatments = cleaned_projects['Treatment Category'].unique()
min_date_planned = cleaned_projects['Date Treatment'].min()

import pickle
from typing import Tuple

# define function for parallelization
def match(cur_start: pd.DataFrame, roadid: str, direction: str, start: int) -> Tuple[pd.Series, pd.Series, pd.Series]:

    # find all projects in that section
    grouped_labels = cleaned_projects[
        (cleaned_projects['RoadID'] == roadid) & \
        (cleaned_projects['Direction'] == direction) & \
        (cleaned_projects['Start'] < start + cur_start.iloc[0]["Length"]) & \
        (cleaned_projects['To Measure'] > start)
    ]

    # find group of train depending on whether there were projects found
    if len(grouped_labels) == 0:
        # Use an arbitrarily late time
        grouped_train = cur_start.sort_values(by=['Date of condition data'], ascending=False).head(3)
    else:
        # Retrieve the date of first treatment
        earliest_project_date = grouped_labels["Date Treatment"].min()
        grouped_train = cur_start[
            (cur_start["Date of condition data"] < earliest_project_date)
        ].sort_values(by=['Date of condition data'], ascending=False).head(3)

        # TODO: think about throwing out these or assigning them as no projects
        if len(grouped_train) == 0:
            return None, None, None

    latest_condition = grouped_train['Date of condition data'].max()

    # compute offset
    if len(grouped_labels) > 0:
        earliest_year = grouped_labels['Date Treatment'].min()
    else:
        earliest_year = grouped_train.iloc[0]['Date of condition data']
    offset_months = ((earliest_year - grouped_train['Date of condition data']) / np.timedelta64(1, 'M')).astype(int)

    # pad offset to 3 and link the month to the index of the train group
    offset_months = list(enumerate(offset_months))
    if len(offset_months) < 3:
        offset_months = [offset_months[0]] * (3 - len(offset_months)) + offset_months

    # flatten train groups
    new_series = {}
    for i, (idx, m) in enumerate(offset_months):
        for col in value_columns:
            new_key = f'{col}|idx={i}'
            new_series[new_key] = grouped_train.iloc[idx][col]
        new_series[f'offset_month|idx={i}'] = m
    else:
        for col in constant_value_columns:
            new_series[col] = grouped_train.iloc[0][col]

    # flatten label groups
    label_mat = make_label_mat(grouped_labels, treatments, latest_condition)
    label_mat = pd.melt(label_mat.reset_index(), id_vars='index').rename(columns={
        'index': 'key_type',
        'variable': 'Treatment Category',
        'value': 'boolean'
    }).set_index(['key_type', 'Treatment Category']).transpose()
    label_mat['no_project_flag'] = 1 if (label_mat.values != 0).sum() == 0 else 0

    new_index = {"RoadID": roadid, "Direction": direction, "Start": start, "Length": grouped_train['Length'].median(), "LengthStd": grouped_train['Length'].std(), "num_treatments": len(grouped_labels), "treatment_idx": grouped_labels.index}
    return new_series, label_mat, new_index

from joblib import Parallel, delayed

job_queue = []
job_pbar = tqdm(desc='Job Queueing', total=cleaned_df.drop_duplicates(["RoadID", "Direction", "Start"]).shape[0])

for road_id in cleaned_df["RoadID"].unique():
    cur_road = cleaned_df[cleaned_df["RoadID"] == road_id]
    for direction in cur_road["Direction"].unique():
        cur_dir = cur_road[cur_road["Direction"] == direction]
        for start in cur_dir["Start"].unique():
            cur_start = cur_dir[cur_dir["Start"] == start]
            job_queue.append(delayed(match)(cur_start, road_id, direction, start))
            job_pbar.update()

import contextlib
import joblib
@contextlib.contextmanager
def tqdm_joblib(tqdm_object: tqdm):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

# append train and labels
with tqdm_joblib(tqdm(desc='Jobs count', total=len(job_queue))):
    results = Parallel(n_jobs=6)(job_queue)

flattened_data = []
flattened_projects = []
flattened_idx = []
for s, l, i in results:
    if s is not None:
        flattened_data.append(s)
        flattened_projects.append(l)
        flattened_idx.append(i)

train_df = pd.DataFrame(flattened_data)
train_labels = pd.concat(flattened_projects, axis=0)
idx_df = pd.DataFrame(flattened_idx)

# no project currently has constant offset, so we replace it with random offset drawn from the set of existing offsets
# for sections with projects
no_project_mask = train_labels.reset_index()["no_project_flag"] == 1
assert set(train_df[no_project_mask]["offset_month|idx=0"].unique()) == set([0])

offset_month_samples = train_df[train_labels.reset_index()["no_project_flag"] == 0]["offset_month|idx=0"]
offset_offset = pd.Series(np.random.choice(offset_month_samples.values, size=len(train_df)))

assert (offset_offset[no_project_mask].index == train_df.loc[no_project_mask].index).all()

old_offset_idx0 = train_df["offset_month|idx=0"]
old_offset_idx1 = train_df["offset_month|idx=1"]
old_offset_idx2 = train_df["offset_month|idx=2"]
old_offsets = pd.concat([old_offset_idx0, old_offset_idx1, old_offset_idx2], axis=1)

train_df.loc[no_project_mask, "offset_month|idx=0"] = train_df.loc[no_project_mask, "offset_month|idx=0"] + offset_offset[no_project_mask]
train_df.loc[no_project_mask, "offset_month|idx=1"] = train_df.loc[no_project_mask, "offset_month|idx=1"] + offset_offset[no_project_mask]
train_df.loc[no_project_mask, "offset_month|idx=2"] = train_df.loc[no_project_mask, "offset_month|idx=2"] + offset_offset[no_project_mask]
train_df.loc[no_project_mask, ["offset_month|idx=0", "offset_month|idx=1", "offset_month|idx=2"]].plot.kde()

old_offsets = old_offsets.assign(offset_offset=offset_offset).assign(offset_applied=no_project_mask)
idx_df["treatment_idx"] = idx_df["treatment_idx"].apply(lambda x: ",".join([str(idx) for idx in x]))

if (DATA_DIR / "processed" / "VIC" / "final").exists() is False:
    (DATA_DIR / "processed" / "VIC" / "final").mkdir(parents=True)

train_df.to_csv(DATA_DIR / "processed" / "VIC" / "final" / "train_all.csv", index=False)
train_labels.to_csv(DATA_DIR / "processed" / "VIC" / "final" / "labels_all.csv", index=False)
old_offsets.to_csv(DATA_DIR / "processed" / "VIC" / "final" / "offsets.csv", index=False)
idx_df.to_csv(DATA_DIR / "processed" / "VIC" / "final" / "match_idx.csv", index=False)
cleaned_df.to_csv(DATA_DIR / "processed" / "VIC" / "final" / "cleaned_condition_data.csv", index=False)
cleaned_projects.to_csv(DATA_DIR / "processed" / "VIC" / "final" / "cleaned_projects.csv", index=False)
