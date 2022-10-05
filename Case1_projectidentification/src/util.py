"""
Template utility functions

Including code written by:
- Catherine Yu (Data Scientist / Engineer)
- David Rawlinson (Lead Data Scientist / Engineer)
- Rafid Morshedi (Senior Data Scientist / Engineer)
"""
import pandas as pd
import pickle

from copy import deepcopy
from typing import Any
from pathlib import Path

def load_data(source: Path, **kwargs) -> pd.DataFrame:
    """
    Return a dataframe loaded from source.\n
    Method is inferred from source name.
    """
    if source.exists() is False:
        raise ValueError(f"{str(source)} does not exist!")
    
    file_type = source.name.split('.')[-1]
    if file_type == 'csv':
        return pd.read_csv(source, **kwargs)
    elif file_type.lower() == 'xlsx':
        return pd.read_excel(source, **kwargs)
    
    raise NotImplementedError(f"Given file extension {file_type} read method is not implemented")

def save_complete_data(df: pd.DataFrame, flag: bool, save_path: Path, save_method: str) -> None:
    """
    Save a well-transformed and processed dataframe into disk. The save path and save method is given in the configuration file.
    """
    if flag is False:
        return
    
    if save_path.parent.exists() is False:
        print("Making save path directory...")
        save_path.parent.mkdir(parents=True)
    
    if save_method == 'save_csv':
        return df.to_csv(str(save_path.resolve()), index=False)
    
    raise NotImplementedError(f"Given method {save_method} is not implemented")

def pickle_data(data: Any, save_path: str, fname: str) -> None:
    """
    Pickle given data into a given save_path folder with a given filename fname.

    Args:
        save_path: must be a folder relative to the root folder
        fname: name of a file
    """
    file_path = save_path / fname

    if save_path.exists() is False:
        save_path.mkdir(parents=True)
    if file_path.exists() is False:
        file_path.touch()
    elif file_path.is_dir():
        raise ValueError(f"Exist a directory with the path {str(save_path.resolve())}")

    pickle.dump(data, open(file_path, 'wb'))

class ConfigNamespace():
    """Wrapper class to allow config to be treated as both a dict and namespace for IDE support"""
    def __init__(self, config_dict: dict):
        """Recursively generate namespaces for nested dicts"""
        self.__dict__.update({
            key: ConfigNamespace(inner) if isinstance(inner, dict) else inner for key, inner in config_dict.items()
        })

    def copy(self) -> "ConfigNamespace":
        """
        Create a deep copy of this namespace
        """
        return ConfigNamespace({
            key: inner.copy() if isinstance(inner, ConfigNamespace) else deepcopy(inner) for key, inner in self.__dict__.items()
        })

    def __setitem__(self, key, item):
        if isinstance(item, dict):
            self.__dict__[key] = ConfigNamespace(item)
        else:
            self.__dict__[key] = item
        
    def __getitem__(self, key):
        return self.__dict__[key]
    
    def items(self):
        return self.__dict__.items()
    
    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()