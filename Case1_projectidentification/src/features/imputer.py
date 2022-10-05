import pandas as pd

class GroupByImputer:
    """
    Impute missing values by forward filling for each group as specified 
    Does not fit on any dataset
    """
    def __init__(self, group_subset: list, sort_subset: list):
        """Initalise the subset by which to group the dataframe and the columns with which to impute"""
        self.group_subset = group_subset # columns to form the group
        self.sort_subset = sort_subset # columns to sort each indivual group
        assert set(sort_subset).intersection(set(group_subset)) == set(), 'Overlapping features between those used to group the data and to sort each group!'
    
    def transform(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        "Forward and Backward fill of the given feature for a dataframe"
        assert col not in self.group_subset and col not in self.sort_subset, 'Column to be imputed is already used for grouping/sorting within each group'
        new_col : pd.Series = df.sort_values(by=self.sort_subset)\
                    .groupby(self.group_subset)[col]\
                    .fillna(method='ffill')\
                    .fillna(method='bfill')
        assert set(df.index) == set(new_col.index) 
        new_col = new_col.reindex(df.index)
        assert df[df[col] != new_col][col].notna().sum() == 0
        df.loc[:, col] = new_col
        return df