import pandas as pd 

from src.config import CONFIG
from ortools.linear_solver import pywraplp
from ortools.linear_solver.pywraplp import Solver, Variable, Constraint, Objective
from typing import Dict, Tuple


def make_variables(df: pd.DataFrame, solver: Solver) -> Dict[int, Variable]:
    return {i: solver.IntVar(0, 1, f'x_{i}') for i in range(len(df))} # indicator for each treatment

def make_budget_constraint(df: pd.DataFrame, solver: Solver, variables: Dict[int, Variable], budget: float) -> Constraint:
    """Create and return constraint for sum of cost <= budget"""
    assert (df[CONFIG['cost']] > 0).all()
    budget_const : Constraint = solver.Constraint(-solver.infinity(), budget)
    for i, var in variables.items():
        budget_const.SetCoefficient(var, df.loc[i, CONFIG['cost']]) 

    return budget_const

def make_objective(df: pd.DataFrame, solver: Solver, variables: Dict[int, Variable], penalties: Dict[str, float]) -> Objective:
    """
    Create and return the objective

    Args:
        df: data to run the optimisation on
        solver: solver to run the optimisation on
        variables: indicator values corresponding to df rows
        penalties: dictionary mapping indicator columns (where true == 1) with penalty value. Positive p generates penalty for true indicator flags, and negatie p generates penalty for negative flag
    """
    obj : Objective = solver.Objective()

    # dynamically adjust penalty heuristic coeff a such that 100a > CONFIG['objective_col_with_treatment'].max() - CONFIG['objective_col_without_treatment].min()
    assert (df[CONFIG['objective_col_with_treatment']] >= df[CONFIG['objective_col_without_treatment']]).all()
    if penalties is not None:
        max_gain = (df[CONFIG['objective_col_with_treatment']] - df[CONFIG['objective_col_without_treatment']]).max()
        pen_coeff = (max_gain + 1) / 100

    # generate objective terms
    for i, var in variables.items():
        # cost term
        with_treat = df.loc[i, CONFIG['objective_col_with_treatment']]
        no_treat = df.loc[i, CONFIG['objective_col_without_treatment']]
        # penalty terms
        pen_terms = []
        for pen_type, p in penalties.items():
            indicator_flag = (df.loc[i, pen_type] == 1) or (df.loc[i, pen_type] == True)
            positive_flag = p > 0
            assert -100 <= p <= 100
            if (positive_flag and indicator_flag) or (not positive_flag and not indicator_flag):
                pen_terms.append(-abs(pen_coeff * p))
            else:
                pen_terms.append(0) 

        # sum of terms
        sum_term = with_treat - no_treat + sum(pen_terms)
        obj.SetCoefficient(var, sum_term)
    obj.SetMaximization()
    return obj

def make_metroreg_constraint(df: pd.DataFrame, solver: Solver, variables: Dict[int, Variable], low: float, high: float):
    """Create and return constraint for metro budget/total budget in low-high range"""
    assert 0 <= low and low <= high and high <= 1, 'Metro/Regional Split value must be within 0-1'

    metroreg_const_low : Constraint = solver.Constraint(0, solver.infinity())
    metroreg_const_high : Constraint = solver.Constraint(-solver.infinity(), 0)

    # a = metro budget, b = reg budget
    # a / (a + b) < h <=> a(1 - h) - hb < 0
    # a / (a + b) > l <=> a(1 - l) - lb > 0 
    for i, var in variables.items():
        indicator_flag = df.loc[i, CONFIG['metro_penalty_col']]
        metroreg_const_low.SetCoefficient(var,
            (indicator_flag - low) * df.loc[i, CONFIG['cost']]
        )
        metroreg_const_high.SetCoefficient(var,
            (indicator_flag - high) * df.loc[i, CONFIG['cost']]
        )

def make_freight_constraint(df: pd.DataFrame, solver: Solver, variables: Dict[int, Variable], low: float, high: float):
    """Create and return constraint for freight budget / all budget in low-high range"""
    assert 0 <= low and low <= high and high <= 1, 'Freight/Non-Freight Split value must be within 0-1'

    freight_const_low : Constraint = solver.Constraint(0, solver.infinity())
    freight_const_high : Constraint = solver.Constraint(-solver.infinity(), 0)

    # a = freight budget, b = non freight budget
    # a / (a + b) < h <=> a(1 - h) - hb < 0
    # a / (a + b) > l <=> a(1 - l) - lb > 0 
    for i, var in variables.items():
        indicator_flag = df.loc[i, CONFIG['freight_penalty_col']]
        freight_const_low.SetCoefficient(var,
            (indicator_flag - low) * df.loc[i, CONFIG['cost']]
        )
        freight_const_high.SetCoefficient(var,
            (indicator_flag - high) * df.loc[i, CONFIG['cost']]
        )

def make_problem_penalties(
    df: pd.DataFrame, 
    budget: float=10000000, 
    penalties: Dict[str, float]={'Metro': 0},
    **kwargs
) -> Tuple[Solver, Dict[int, Variable], Objective]:
    """Make the formulation and return the following in order: the solver the problem was mapped on, assignment variables, the objective"""
    # Create the mip solver with the SCIP backend.
    solver : Solver = pywraplp.Solver.CreateSolver('SCIP')

    var_dict = make_variables(df, solver)
    make_budget_constraint(df, solver, var_dict, budget)
    objective = make_objective(df, solver, var_dict, penalties)

    return solver, var_dict, objective

def make_problem_hardconstraint(
    df: pd.DataFrame, 
    budget: float=10000000, 
    metro_range: tuple=(0, 1), 
    freight_range: tuple=(0, 1), 
    **kwargs
) -> Tuple[Solver, Dict[int, Variable], Objective]:
    """Make the formulation and return the following in order: the solver the problem was mapped on, assignment variables, the objective"""
    # Create the mip solver with the SCIP backend.
    solver : Solver = pywraplp.Solver.CreateSolver('SCIP')

    var_dict = make_variables(df, solver)
    make_budget_constraint(df, solver, var_dict, budget)
    make_metroreg_constraint(df, solver, var_dict, metro_range[0], metro_range[1])
    make_freight_constraint(df, solver, var_dict, freight_range[0], freight_range[1])
    objective = make_objective(df, solver, var_dict, penalties={})

    return solver, var_dict, objective