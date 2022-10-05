"""
Implement different strategies for controlling inner optimisation routine
"""

import pandas as pd
from collections import deque
from ortools.linear_solver.pywraplp import Solver, Variable, Objective
from ortools.linear_solver import linear_solver_pb2
from typing import Dict, Tuple, List, Callable

SolverTuple = Tuple[int, Solver, Dict[int, Variable], Objective]
MakeCallable = Callable[..., Tuple[Solver, Dict[int, Variable], Objective]]


def optimal_solve(df: pd.DataFrame, method: MakeCallable, timeout: int=None, **kwargs) -> SolverTuple:
    """
    Solve problems optimally with an optional time out in miliseconds
    
    Args:
        df: the dataframe to run optimisation on
        timeout: maximum wait time in miliseconds

    Returns: a tuple containing in order
        status: status of the solver. Can be: 0 (optimal result), 1 (feasible or stopped by time limit), 2 (proven infeasible), 3 (unbounded), 4 (error), 6 (not solved)
        solver: the solver
        var_dict: dictionary mapping integer index of the dataframe to indicator variable
        objective: the objective
    """
    solver, var_dict, objective = method(df, **kwargs)
    verifications = []

    if timeout:
        solver.set_time_limit(timeout)

    status = solver.Solve()
    verifications.append(solver.VerifySolution(1e-7, False))
    return (status, solver, var_dict, objective), verifications

# https://github.com/google/or-tools/issues/1469
def set_hint_for_all_variables(solver: Solver):
    """Set current solution as hint"""
    # Get all variable assignments
    solution = linear_solver_pb2.MPSolutionResponse()
    solver.FillSolutionResponseProto(solution)

    # Get model and set solution hint
    model = linear_solver_pb2.MPModelProto()
    solver.ExportModelToProto(model)
    model.solution_hint.Clear()
    for index, value in enumerate(solution.variable_value):
        model.solution_hint.var_index.append(index)
        model.solution_hint.var_value.append(value)
    solver.LoadModelFromProto(model)


def early_stop(df: pd.DataFrame, method: MakeCallable, timeout: int=60000, patience: int=1, **kwargs) -> Tuple[SolverTuple, List[bool]]:
    """
    Strategy for early stopping if performance does not improve

    Args:
        df: the dataframe to run optimisation on
        count_patience: number of time outs before early stopping is done

    Returns: a tuple containing in order
        status: status of the solver. Can be: 0 (optimal result), 1 (feasible or stopped by time limit), 2 (proven infeasible), 3 (unbounded), 4 (error), 6 (not solved)
        solver: the solver
        var_dict: dictionary mapping integer index of the dataframe to indicator variable
        objective: the objective
        verification_output: list of bool output from solver verification
    """

    solver, var_dict, objective = method(df, **kwargs)

    solver.set_time_limit(timeout) # 0.1s time out
    last_metrics = deque() # keep record of previous results
    verification_output = []

    i = 1
    while True:
        status = solver.Solve()
        verification_output.append(solver.VerifySolution(1e-7, False))
        if status == solver.OPTIMAL:
            break # optimal solution found
        
        obj_val = solver.Objective().Value()

        # early stopping logic
        if len(last_metrics) < patience:
            if len(last_metrics) > 0:
                # solution should not gets worse with more iteration
                assert (obj_val >= last_metrics[-1]) or abs(obj_val - last_metrics[-1]) < 1e-6
            last_metrics.append(obj_val)
        else:
            removed = last_metrics.popleft()
            if len(last_metrics) > 0:
                assert (obj_val >= last_metrics[-1]) or abs(obj_val - last_metrics[1]) < 1e-6
            else:
                assert (obj_val >= removed) or abs(obj_val - removed) < 1e-6
            last_metrics.append(obj_val)

            if abs(removed - obj_val) / (removed + 1e-8) < 0.01: # if improvement is less than 1%, break
                break
            
        solver.set_time_limit(i * timeout) # extend by 10s

    return (status, solver, var_dict, objective), verification_output
