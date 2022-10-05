"""Utility methods for testing"""

import pandas as pd

from src.config import CONFIG
from typing import Dict
from data import DATA_DIR
from ortools.linear_solver.pywraplp import Solver, Variable
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

def get_objective_value(
    df: pd.DataFrame, 
    selection: pd.Series, 
) -> Dict[str, float]:
    """Compute LoS, Metro Percentage, and Freight Percentage from selection vector"""
    selected = df[selection == 1]
    rejected = df[selection == 0]
    total_cost = selected[str(CONFIG['cost'])].sum()
    delta_los = selected[CONFIG['objective_col_with_treatment']].sum() + rejected[CONFIG['objective_col_without_treatment']].sum()
    abs_los = selected[CONFIG['los_after_with_treatment']].sum()  + rejected[CONFIG['los_after_without_treatment']].sum()

    if total_cost == 0:
        return {
            'metro_perc': 0,
            'freight_perc': 0,
            'dLoS': delta_los,
            'Absolute_LoS': abs_los,
            'total_cost': 0,
        }
    else:
        selected_metro = df.loc[(selection == 1) & (df[CONFIG['metro_penalty_col']] == 1), :]
        selected_freight = df.loc[(selection == 1) & (df[CONFIG['freight_penalty_col']] == 1), :]
        return {
            'metro_perc': selected_metro[CONFIG['cost']].sum() / total_cost,
            'freight_perc': selected_freight[CONFIG['cost']].sum() / total_cost,
            'dLoS': delta_los,
            'Absolute_LoS': abs_los,
            'total_cost': total_cost
        }

def var_dict_to_series(var_dict: Dict[int, Variable]) -> pd.Series:
    return pd.Series({idx: var.solution_value() for idx, var in var_dict.items()})

def make_solve_problem(solver: Solver, var_dict: Dict[int, Variable]):
    """Make the problem and solve it"""
    solver.Solve()
    return pd.Series({idx: var.solution_value() for idx, var in var_dict.items()}) 

def load_data() -> pd.DataFrame:
    df = pd.read_csv(Path('.').resolve().parent / 'data' / 'NLTP_Unlimited_dTAGTL.csv')
    df.columns = df.columns.str.strip()

    # Mark Freight sections.
    df.loc[:, 'Freight'] = df['group_desc'].str.contains(r'(?:High Volume|National|Regional)', regex=True)

    # Correct rows with a committed treatment.
    com_trt_mask = df[str(CONFIG["committed_treatment_col"])].notna()
    df.loc[com_trt_mask, "DNPCI_Before"] = df.loc[com_trt_mask, "PCI_Before"]
    df.loc[com_trt_mask, "DNPCI_After"] = df.loc[com_trt_mask, "PCI_After"]

    # Calculate lane-length weighted LoS heuristic (normalised PCI).
    df[CONFIG['lane_len_col']] = df[CONFIG['lane_col']] * df[CONFIG['len_col']]
    max_col = df[["PCI_Before", "PCI_After", "DNPCI_Before", "DNPCI_After"]].max().idxmax()
    minmax_scaler = MinMaxScaler(feature_range=(0, 100))
    minmax_scaler.fit((df[max_col] * df[CONFIG["lane_len_col"]]).values.reshape(-1, 1))
    df["nPCI_Before"] = minmax_scaler.transform((df["PCI_Before"] * df[CONFIG["lane_len_col"]]).values.reshape(-1, 1))
    df["nPCI_After"] = minmax_scaler.transform((df["PCI_After"] * df[CONFIG["lane_len_col"]]).values.reshape(-1, 1))
    df["nDNPCI_Before"] = minmax_scaler.transform((df["DNPCI_Before"] * df[CONFIG["lane_len_col"]]).values.reshape(-1, 1))
    df["nDNPCI_After"] = minmax_scaler.transform((df["DNPCI_After"] * df[CONFIG["lane_len_col"]]).values.reshape(-1, 1))
    df["n$200MPCI_Before"] = minmax_scaler.transform((df["$200MPCI_Before"] * df[CONFIG["lane_len_col"]]).values.reshape(-1, 1))
    df["n$200MPCI_After"] = minmax_scaler.transform((df["$200MPCI_after"] * df[CONFIG["lane_len_col"]]).values.reshape(-1, 1))

    # Calculate deltas.
    df.loc[:, 'DNdPCI'] = df['DNPCI_After'] - df['DNPCI_Before']
    df.loc[:, 'nd$200MPCI'] = df['n$200MPCI_After'] - df['n$200MPCI_Before']
    df.loc[:, CONFIG['objective_col_with_treatment']] = df['nPCI_After'] - df['nPCI_Before']
    df.loc[:, CONFIG['objective_col_without_treatment']] = df['nDNPCI_After'] - df['nDNPCI_Before']

    # We are only interested in rows with a treatment.
    df = df[(df[CONFIG["objective_col_with_treatment"]] > df[CONFIG['objective_col_without_treatment']]) & (df['Cost'] > 0)].reset_index(drop=True)
    return df
