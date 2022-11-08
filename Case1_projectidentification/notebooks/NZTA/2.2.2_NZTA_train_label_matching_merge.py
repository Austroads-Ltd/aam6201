from os import cpu_count
import pandas as pd
import numpy as np
import src.util as util

from src.features import preprocessing
from typing import Tuple, List
from copy import deepcopy

# load data
from src import DATA_DIR
DATASET = 'NZTA'
experiment_suffix = 'nzta_final'
train_df = util.load_data(source=DATA_DIR / 'interim' / DATASET / f'train_processed{("_" + experiment_suffix) if experiment_suffix else ""}.csv')
valid_df = util.load_data(source=DATA_DIR / 'interim' / DATASET / f'valid_processed{("_" + experiment_suffix) if experiment_suffix else ""}.csv')

experiment_suffix = 'nzta_final'
projects = pd.read_excel(DATA_DIR / 'raw' / 'NZTA' / 'Raw Data' / 'NZ_SH_FWP21_24 by NOC.xlsx', sheet_name='10Yr')

# clean projects
project_regex = {
    '^AC_?[0-9]*_?(PMB|)_?L?[1-2]?$': 'Resurfacing_AC',
    '^.*OGPA.*$': 'Resurfacing_AC',
    '^(PROJ_)?TAC(_SH15)?$': 'Resurfacing_AC',
    '^EPA$': 'Resurfacing_AC',
    '^.*UTA.*$': 'Resurfacing_AC',
    '^SMA.*$': 'Resurfacing_AC',
    '^.*[Hh](EA)?[Vv][Yy].*$': 'Major Patching',
    '^.*[Mm]aj.*$': 'Major Patching',
    '^OLAY.*$': 'Rehabilitation',
    '^.*[Ss][Tt][Aa][Bb].*$': 'Rehabilitation',
    '^RH(FBS)?.*$': 'Rehabilitation',
    'AWT': 'Rehabilitation',
    '^SAC.*$': 'Rehabilitation',
    'RECON': 'Rehabilitation',
    '^RECY.*$': 'Rehabilitation',
    '^RMUP.*$': 'Rehabilitation',
    '^RS?[2-9]+.*$': 'Resurfacing_SS',
    '^RS(B|M|S)?(_SH15)?$': 'Resurfacing_SS',
    '^FBSMUP$': 'Rehabilitation',
    '^(FAB|S)?(SEAL|VF).*$': 'Resurfacing_SS',
    '^CS$': 'Resurfacing_SS',
    '^(min)?SecondCoat.*$': 'Resurfacing_SS',
    '^CAPE$': 'Resurfacing_SS',
    '^COMBI$': 'Resurfacing_SS',
    '^(PROJ_)?G.*$': 'Resurfacing_SS',
    '^[Rr]E?[Jj]U?[Vv][Nn]?.*$': 'Retexturing',
    '^S[Ll]{1}u?[Rr]+[Yy]{1}$': 'Regulation',
}

# check whether regex overlap and cause misattribution
import warnings
warnings.filterwarnings('ignore')

duplicated = {}
for yr in range(2021, 2031):
    matches = []
    for regex in project_regex:
        matches.append(projects[f'treat_{yr}'].str.contains(regex, regex=True))

    matches = pd.concat(matches, axis=1)
    overlap_count = len(matches[matches.sum(axis=1) > 1]) 

    duplicated[yr] = set(projects[matches.sum(axis=1) > 1][f'treat_{yr}'])

warnings.filterwarnings('always')

# replace with regex
regex_projects = projects.copy()

old_shape = regex_projects.shape[0]
regex_projects = regex_projects.dropna(
    subset=[f'treat_{yr}' for yr in range(2021, 2031)],
    how='all'
)
new_shape = regex_projects.shape[0]

for yr in range(2021, 2031):
    regex_projects.loc[:, f'treat_{yr}'] =\
        projects[f'treat_{yr}'].replace(project_regex, regex=True)

old_shape = regex_projects.shape[0]
keep = regex_projects[[f'treat_{yr}' for yr in range(2021, 2031)]].isin({'Regulation', 'Rehabilitation', 'Resurfacing_AC', 'Resurfacing_SS', 'Retexturing', 'Regulation', 'Major Patching'})
# values that are not nans but not in general categories either
invalid = ((~keep) & regex_projects[[f'treat_{yr}' for yr in range(2021, 2031)]].notna()).any(axis=1)
filtered_projects = regex_projects[keep.any(axis=1) & ~invalid]
new_shape = filtered_projects.shape[0]

# check final categories
from collections import Counter
categories = Counter()
all_vals = filtered_projects\
    [[f'treat_{yr}' for yr in range(2021, 2031)]].values

for i in range(all_vals.shape[0]):
    categories.update(all_vals[i])

# cast id to standard form
cleaned_projects = filtered_projects.drop(columns=['sectionID'])
cleaned_projects = cleaned_projects.rename(columns={
    'sectionName': 'RoadID',
    'locFrom': 'Start',
    'locTo': 'End'
})

cleaned_projects['Date Planned'] = pd.to_datetime('2020-03-01', format='%Y-%m-%d')

old_shape = cleaned_projects.shape
cleaned_projects = cleaned_projects.dropna(
    subset=[f'treat_{yr}' for yr in range(2021, 2031)],
    how='all'
)

cleaned_projects = cleaned_projects[['RoadID', 'Start', 'End', 'Date Planned'] + [f'treat_{yr}' for yr in range(2021, 2031)]]
cleaned_projects = cleaned_projects.melt(id_vars=['RoadID', 'Start', 'End', 'Date Planned'], value_vars=[f'treat_{yr}' for yr in range(2021, 2031)], value_name='Treatment Category', var_name='Date Treatment')
cleaned_projects = cleaned_projects.dropna(subset=['Treatment Category'])
cleaned_projects.loc[:, 'Date Treatment'] = pd.to_datetime(cleaned_projects['Date Treatment'].str.split('_', expand=True)[1], format='%Y')
cleaned_projects = cleaned_projects.drop_duplicates()


cleaned_projects.to_csv(DATA_DIR / 'interim' / DATASET / 'cleaned_projects.csv', index=False)


train_df.loc[:, 'Date of condition data'] = train_df['Date of condition data'].astype(np.datetime64)
valid_df.loc[:, 'Date of condition data'] = valid_df['Date of condition data'].astype(np.datetime64)

def make_label_mat(grouped_labels: pd.DataFrame, treatments: list) -> pd.DataFrame:
    label_mat = pd.DataFrame(columns=treatments, index=[
        'Treatment within 1 year', # 1 Year
        'Treatment between 1 to 3 years', # 2 - 3 Year
        'Treatment between 3 to 5 years', # 4 - 5 Year
        'Treatment between 5 to 10 years', # 6 - 10 Year
        'Treatment between 10 to 30 years', # 10 - 30 Years
    ])
    label_mat.loc[:, :] = 0
    if len(grouped_labels) == 0:
        return label_mat

    for i, treatment in enumerate(treatments):
        category_labels = grouped_labels[grouped_labels['Treatment Category'] == treatment]
        if len(category_labels) == 0:
            continue
        
        year_offset = ((category_labels['Date Treatment'] - category_labels['Date Planned']) / np.timedelta64(1, 'Y'))

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


def make_train_label(index: dict, 
    cleaned_projects: pd.DataFrame, 
    yearless_dfs: pd.DataFrame, 
    year_dfs: pd.DataFrame, 
    all_planned_dates: np.ndarray, 
    unique_treatments: List[str]) -> Tuple[List[pd.Series], List[pd.Series], List[pd.Series]]:
    
    # find group of projects which intersect with index
    grouped_labels = cleaned_projects[
        (cleaned_projects['RoadID'] == index['RoadID']) & \
        (cleaned_projects['Start'] < index['End']) & \
        (cleaned_projects['End'] > index['Start'])
    ]

    samples = []
    labels = []
    indices = []
    planned_dates = grouped_labels['Date Planned'].unique() if len(grouped_labels) > 0 else [np.random.choice(all_planned_dates)]
    # each plan date is a different decision making action on the group of data associated with the index
    for planned_date in planned_dates:

        # find data which preceed plan date
        train_with_year_dfs = []
        for cleaned_df in year_dfs:
            grouped_train = cleaned_df[
                (cleaned_df['RoadID'] == index['RoadID']) & \
                (cleaned_df['Start'] == index['Start']) & \
                (cleaned_df['Date of condition data'] < planned_date)
            ].sort_values(by='Date of condition data', ascending=False).head(3)
            assert len(grouped_train) > 0 # spatial index lacks information on one of the dataframe
            train_with_year_dfs.append(grouped_train)

        flattened_vector = {}
        for i, grouped_train in enumerate(train_with_year_dfs):
            # compute offset
            offset_months = ((planned_date - grouped_train['Date of condition data']) / np.timedelta64(1, 'M')).astype(int)
            offset_months = list(enumerate(offset_months)) # pad offset to 3 and link the month to the index of the train group
            if len(offset_months) < 3:
                offset_months = [offset_months[0]] * (3 - len(offset_months)) + offset_months
            # flatten train groups
            new_series = {}
            for j, (idx, m) in enumerate(offset_months):
                for col in grouped_train.columns:
                    if col in ['RoadID', 'Start', 'Direction', 'End', 'Date of condition data']:
                        continue
                    new_key = f'{col}_df{i}|idx={j}'
                    new_series[new_key] = grouped_train.iloc[idx][col]
                new_series[f'offset_month_df{i}|idx={j}'] = m
            flattened_vector.update(new_series)
        
        # get train that does not care about time
        for i, df in enumerate(yearless_dfs):
            new_series = df[
                (df['RoadID'] == index['RoadID']) &\
                (df['Start'] == index['Start'])
            ]
            assert len(new_series) == 1
            new_series = new_series.drop(columns=[col for col in ['RoadID', 'Start', 'End', 'Direction'] if col in new_series.columns]).iloc[0, :].to_dict()
            flattened_vector.update(new_series)

        # flatten label groups
        label_mat = make_label_mat(grouped_labels[grouped_labels['Date Planned'] == planned_date], unique_treatments)
        label_mat = pd.melt(label_mat.reset_index(), id_vars='index').rename(columns={
            'index': 'key_type',
            'variable': 'Treatment Category',
            'value': 'boolean'
        }).set_index(['key_type', 'Treatment Category']).transpose()
        label_mat['no_project_flag'] = 1 if (label_mat.values != 0).sum() == 0 else 0

        # append train and labels
        samples.append(pd.Series(flattened_vector))
        labels.append(label_mat)
        # append planned date to index and saves it
        new_index = deepcopy(index)
        new_index['Date Planned'] = planned_date
        indices.append(pd.Series(new_index))
        
    return samples, labels, indices

def make_train_label_chunk(index_chunk: pd.DataFrame, *args, **kwargs):
    flattened_train, flattened_labels, flattened_indices = [], [], []
    for _, index in index_chunk.iterrows():
        new_train, new_labels, new_indices = make_train_label(index.to_dict(), *args, **kwargs)
        flattened_train.extend(new_train)
        flattened_labels.extend(new_labels)
        flattened_indices.extend(new_indices)
    return flattened_train, flattened_labels, flattened_indices
        

from src import DATA_DIR
from joblib import Parallel, delayed
import time

treatments = cleaned_projects['Treatment Category'].unique()
all_planned_dates = cleaned_projects['Date Planned'].values


for df_type, cleaned_df, num in [('train', train_df, len(train_df.drop_duplicates(['RoadID', 'Start']))), ('valid', valid_df, len(valid_df.drop_duplicates(['RoadID', 'Start'])))]:

    indexed_df = cleaned_df.drop_duplicates(['RoadID', 'Start'])\
        [[col for col in ['RoadID', 'Direction', 'Start', 'End', 'Date of condition data'] if col in cleaned_df.columns]]

    flattened_data = []
    flattened_projects = []
    flattened_indices = []
    
    start = time.time()
    tasks_count = cpu_count() 
    results = Parallel(n_jobs=tasks_count)\
        (delayed(make_train_label_chunk)(
            chunk, cleaned_projects, [], [cleaned_df], all_planned_dates, treatments
        ) for chunk in np.array_split(indexed_df.sample(num), tasks_count)
    )

    # collect results
    for (train, label, indices) in results:
        flattened_data.extend(train)
        flattened_projects.extend(label)
        flattened_indices.extend(indices)

    df = pd.concat(flattened_data, axis=1).transpose()
    labels = pd.concat(flattened_projects, axis=0)
    indices = pd.concat(flattened_indices, axis=1).transpose()

    df.to_csv(DATA_DIR / 'processed' / DATASET / f'{df_type}_flattened_data{"_" + experiment_suffix if experiment_suffix else ""}.csv', index=False)
    labels.to_csv(DATA_DIR / 'processed' / DATASET / f'{df_type}_flattened_labels{"_" + experiment_suffix if experiment_suffix else ""}.csv', index=False)
    indices.to_csv(DATA_DIR / 'processed' / DATASET / f'{df_type}_flattened_index{"_" + experiment_suffix if experiment_suffix else ""}.csv', index=False)


train_flattened_df = util.load_data(DATA_DIR / 'processed' / DATASET / f'train_flattened_data{("_" + experiment_suffix) if experiment_suffix else ""}.csv')
valid_flattened_df = util.load_data(DATA_DIR / 'processed' / DATASET / f'valid_flattened_data{("_" + experiment_suffix) if experiment_suffix else ""}.csv')
train_flattened_labels = util.load_data(DATA_DIR / 'processed' / DATASET / f'train_flattened_labels{("_" + experiment_suffix) if experiment_suffix else ""}.csv', header=[0, 1])
valid_flattened_labels = util.load_data(DATA_DIR / 'processed' / DATASET / f'valid_flattened_labels{("_" + experiment_suffix) if experiment_suffix else ""}.csv', header=[0, 1])

offset_scale_config = {
    'target': None,
    'preprocessing': {
        'normalizing': {
            'feature_list': [col for col in train_flattened_df.columns if col.startswith('offset_month_df')],
            'leave_out': False
        }
    }
}

offset_scaler = preprocessing.fit_scaler(train_flattened_df, offset_scale_config)
scaled_train_flattened_df = preprocessing.scale(train_flattened_df, offset_scaler, offset_scale_config)
scaled_valid_flattened_df = preprocessing.scale(valid_flattened_df, offset_scaler, offset_scale_config)

# error checking
scaled_idx0 = scaled_train_flattened_df['offset_month_df0|idx=0']
idx0 = train_flattened_df['offset_month_df0|idx=0']
assert ((idx0 - idx0.mean()) / idx0.std() - scaled_idx0).abs().max() < 5e-4

# overwrite unscaled data 
scaled_train_flattened_df.to_csv(DATA_DIR / 'processed' / DATASET / f'train_flattened_data{("_" + experiment_suffix) if experiment_suffix else ""}.csv', index=False)
scaled_valid_flattened_df.to_csv(DATA_DIR / 'processed' / DATASET / f'valid_flattened_data{("_" + experiment_suffix) if experiment_suffix else ""}.csv', index=False)

# get without offset
scaled_train_flattened_df_no_offset = scaled_train_flattened_df[[col for col in scaled_train_flattened_df.columns if (('idx=0' in col) and ('offset' not in col))]]
scaled_valid_flattened_df_no_offset = scaled_valid_flattened_df[[col for col in scaled_valid_flattened_df.columns if (('idx=0' in col) and ('offset' not in col))]]
scaled_train_flattened_df_no_offset.to_csv(DATA_DIR / 'processed' / DATASET / f'train_flattened_data{("_" + experiment_suffix) if experiment_suffix else ""}_no_offset.csv', index=False)
scaled_valid_flattened_df_no_offset.to_csv(DATA_DIR / 'processed' / DATASET / f'valid_flattened_data{("_" + experiment_suffix) if experiment_suffix else ""}_no_offset.csv', index=False)
