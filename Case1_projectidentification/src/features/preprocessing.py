"""
Library methods for preprocessing:
- Imputing missing values according to certain strategies
- One hot encoding of specified variables
- Normalization of specified variables according to settings

Contains code written by
- Catherine Yu (Data Scientist / Engineer)
- David Rawlinson (Lead Data Scientist / Engineer)
- Rafid Morshedi (Senior Data Scientist / Engineer)
"""
import pandas as pd
import numpy as np

from typing import Dict, Any, List
from sklearn.preprocessing._encoders import _BaseEncoder # for type hinting
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from src.util import ConfigNamespace
from src.features.imputer import GroupByImputer

# TODO: generalise strategy
def fit_imputer(df: pd.DataFrame, config: ConfigNamespace) -> Dict[Any, SimpleImputer]:
    """
    Fit a median imputer on each column of a given dataframe
    """
    imputer_dict = {}
    features = config['preprocessing']['imputing']['feature_list']
    leave_out = config['preprocessing']['imputing']['leave_out']
    if leave_out is True:
        df = df.drop(columns=features)
    else:
        df = df[features].copy()

    for col in df.select_dtypes(include='number').columns:
        imr = SimpleImputer(missing_values=np.nan, strategy='median')
        try:
            imr = imr.fit(df[[col]])
            imputer_dict[col] = imr
        except Exception:
            print('Error imputing values for col:', col)
    return imputer_dict

def impute(imputer_dict: Dict[Any, SimpleImputer], df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a dictionary mapping a column to an imputer, impute missing values on columns of a given dataframe
    """
    copy_df = df.copy()
    for col, imr in imputer_dict.items():
        try:
            copy_df.loc[:, col] = imr.transform(copy_df[col].to_numpy().reshape(-1, 1)).ravel()
        except KeyError:
            print(f"Cannot impute column {col}; column does not exist in given dataframe.")
    return (copy_df)

def groupby_impute(df: pd.DataFrame, config: ConfigNamespace) -> pd.DataFrame:
    """
    Impute missing values by forward and backward filling through groups of samples
    """
    df = df.copy()
    features = config['preprocessing']['imputing']['groupby_first']['feature_list']
    group_subset = config['preprocessing']['imputing']['groupby_first']['groupby_subset']
    sort_subset = config['preprocessing']['imputing']['groupby_first']['sort_subset']
    subset = set(group_subset).union(sort_subset)
    imputer = GroupByImputer(group_subset, sort_subset)

    for col in features:
        if (col in df.columns) and (col not in subset):
            df.loc[:, col] = imputer.transform(df, col)
        elif col not in df.columns:
            print(f"Cannot impute missing values for {col} with groups; it is not in the data.")
        else:
            print(f"Cannot impute {col} with groups; it is used for grouping or sorting each groups.")

    return df

# TODO: generalise strategy beyond one hot
def get_categorical_features(df: pd.DataFrame, config: ConfigNamespace) -> pd.DataFrame:
    """
    Given a list of categorical columns in the configuration, returns a dataframe containing:\n
    \tThe name of those columns\n
    \tThe number of unique values to be processed, the rest are treated as 'others'\n
    \tThe type of encoding to be performed.
    """
    return_df = {'Features': [], 'No. of feature cols': [], 'Type': [], 'Keep Other': []}
    # collect features and the corresponding numbers of values to be kept after encoding. The rest will be treated as 'others'
    feature_list : List[str]
    enc_count_lst : List[int]
    keep_other_lst : List[bool]
    feature_list = [(feature, 'OneHot') for feature in config['preprocessing']['cat_encoding']['OneHot']['cat_col_list']]
    feature_list.extend([(feature, 'Ordinal') for feature in config['preprocessing']['cat_encoding']['Ordinal']['cat_col_list']])
    enc_count_lst = config['preprocessing']['cat_encoding']['OneHot']['encoding_count']
    enc_count_lst.extend(config['preprocessing']['cat_encoding']['Ordinal']['encoding_count'])
    keep_other_lst = config['preprocessing']['cat_encoding']['OneHot']['keep_other']
    keep_other_lst.extend(config['preprocessing']['cat_encoding']['Ordinal']['keep_other'])

    for i, (feature, enc_type) in enumerate(feature_list):
        return_df['Features'].append(feature)
        if enc_count_lst[i]:
            return_df['No. of feature cols'].append(min(enc_count_lst[i], df[feature].nunique()))
        else:
            return_df['No. of feature cols'].append(df[feature].nunique())
        return_df['Keep Other'].append(keep_other_lst[i])
        return_df['Type'].append(enc_type)

    return pd.DataFrame(data=return_df) 

def get_categorical_encoding(df: pd.DataFrame, config: ConfigNamespace) -> dict:
    """
    Return a dictionary containing the names of categorical features and their corresponding encoder
    """
    cat_feat_df = get_categorical_features(df, config)
    cat_feat_names = set(cat_feat_df['Features'])
    feature_encoding = {}
    for col in df.columns:
        if str(col) in cat_feat_names:
            enc_dict = {}
            enc_type = cat_feat_df.loc[cat_feat_df['Features']==col]['Type'].item()
            # get the unique values to be kept
            sorted_list = df[col].value_counts().sort_values(ascending=False).index.to_list()
            num = cat_feat_df.loc[cat_feat_df['Features']==col]['No. of feature cols'].item()
            keep = sorted_list[:num]       

            if enc_type == 'OneHot':
                #Apply one hot encoder 
                encoder = MultiLabelBinarizer()
            elif enc_type == 'Ordinal':
                # Apply ordinal encoding
                encoder = OrdinalEncoder() 

            enc_dict['encoder'] = encoder
            enc_dict['fitted'] = False
            enc_dict['keep'] = keep
            enc_dict['keep_other'] = cat_feat_df.loc[cat_feat_df['Features'] == col]['Keep Other'].item()
            enc_dict['enc_type'] = enc_type
            feature_encoding[str(col)] = enc_dict # sorted_list

    return feature_encoding

def encode_categorical_features(df: pd.DataFrame, config: dict, feature_encoding: dict) -> pd.DataFrame:
    """
    Given the encodings of features, perform one hot encoding on the inputted argument
    """
    copy = df.copy()
    cat_feat_df = get_categorical_features(copy, config) # get categorical features and count of unique values
    cat_feat_set = set(cat_feat_df['Features'])

    for col in copy.columns:
        if str(col) in cat_feat_set:
            enc = feature_encoding[col]
            
            # Transform raw values to top-num constants and 'other'
            copy.loc[~df[col].isin(enc['keep']), col] = 'other_flag!' # set all values beyond the `num` most common to others. Add flag in case dataset also contains 'other'

            def enlist_items(lst):
                # [a, b, c] -> [[a], [b], [c]]
                return list(map(lambda el:[el], lst))

            # Fit the encoder and transform output values
            encoder : _BaseEncoder = enc['encoder']
            col_values = copy[col].astype(str).tolist()
            if enc['fitted'] is False:
                encoder.fit(enlist_items(col_values))
                enc['fitted'] = True

            # Transform 
            tx = encoder.transform(enlist_items(col_values)) # transform result
            if enc['enc_type'] == 'OneHot': 
                keep_other = enc['keep_other']
                # populate new columns for one hot encoded feature
                col_names = [] # new column names
                for class_name in encoder.classes_:
                    prefix_class_name = col+'_'+class_name
                    col_names.append(prefix_class_name)
                df_temp = pd.DataFrame(tx, columns=col_names, index=df.index)
                copy = pd.concat([copy, df_temp], axis = 1)

                if len(enc['keep']) >= 1 and keep_other is False:
                    try:
                        copy = copy.drop([col+'_'+'other_flag!'], axis=1)
                    except KeyError: # all values encoded, no other flag is here
                        pass
                else:
                    try:
                        copy = copy.rename(columns={f'{col}_other_flag!': f'{col}_Other'})
                    except KeyError: # all values encoded, no other flag is here
                        pass
                copy = copy.drop([col], axis=1)

            if enc['enc_type'] == 'Ordinal':
                copy.loc[:, col] = [val[0] for val in tx]

    
    return copy

def fit_scaler(df: pd.DataFrame, config: ConfigNamespace) -> Dict[str, StandardScaler]:
    """
    Fit a standard scaler on each column of a given dataframe and return a dictionary of scalers
    """ 
    feature_list = config['preprocessing']['normalizing']['feature_list']
    leave_out = config['preprocessing']['normalizing']['leave_out']

    if leave_out:
        exclude = feature_list 
    else:
        exclude = list(set(df.columns) - set(feature_list))

    exclude += [config['target']] if config['target'] else []
    exclude = set(exclude).union(df.select_dtypes('object').columns)

    scaler_dict = {}
    for feature in set(df.columns) - exclude:   
        scaler = StandardScaler().fit(df[feature].to_numpy().reshape(-1, 1))
        scaler_dict[feature] = scaler
    return scaler_dict

def scale(df: pd.DataFrame, scaler_dict: Dict[str, StandardScaler], config: ConfigNamespace) -> pd.DataFrame:
    """
    Scale a given dataframe with a given fitted scaler
    """
    copy_df = df.copy()
    for feature, scaler in scaler_dict.items():
        if feature not in df.columns:
            print(f"Cannot scale {feature}; feature not in dataframe.")
        elif df[feature].dtype.kind not in 'biuf':
            print(f"Cannot scale {feature}; feature is not numeric.")
        else:
            copy_df.loc[:, feature] = scaler.transform(df[feature].to_numpy().reshape(-1, 1))
    return copy_df