import pandas as pd

from data import DATA_DIR
from typing import List, Tuple

def make_constraint(treatment: str, feature_names: List[str]) -> Tuple[int]:
    relationships = pd.read_csv(DATA_DIR.parent / 'references' / 'known_relationships.csv')
    try:
        relationships = relationships[['category', treatment.lower()]]
        known_categories = set(relationships['category'])
        assert len(known_categories) == relationships['category'].nunique()
    except KeyError:
        raise ValueError("Treatment must be one of ['Resurfacing_SS', 'Resurfacing_AC', 'Rehabilitation']")
    
    constraint_tuple = []
    for feature in feature_names:
        matches = [known for known in known_categories if known in feature.lower()]
        if len(matches) > 0:
            vals = set(relationships.loc[relationships['category'].isin(matches), treatment.lower()])
            assert len(vals) == 1
            constraint_tuple.append(vals.pop())
        else:
            constraint_tuple.append(0) # no constraint

    return tuple(constraint_tuple)

