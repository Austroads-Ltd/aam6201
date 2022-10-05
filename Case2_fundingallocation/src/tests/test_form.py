import pandas as pd
import unittest

from ortools.linear_solver.pywraplp import Objective, Solver, Variable
from typing import Dict, Callable, Tuple

from src.config import CONFIG 
from src.forms.ip_form import make_problem_penalties, make_problem_hardconstraint
from src.util import get_objective_value, load_data, make_solve_problem


PROBLEM_FORMS : Dict[str, Callable[..., Tuple[Solver, Dict[int, Variable], Objective]]] = {
    'integer-programming-penalties': make_problem_penalties,
    'integer-programming-hardconstraint': make_problem_hardconstraint
}

class TestFormulation(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.toy_data = load_data()
        cls.metro_budget = cls.toy_data[cls.toy_data[CONFIG['metro_penalty_col']] == 1][CONFIG['cost']].sum()
        cls.reg_budget = cls.toy_data[cls.toy_data[CONFIG['metro_penalty_col']] == 0][CONFIG['cost']].sum()
        cls.freight_budget = cls.toy_data[cls.toy_data[CONFIG['freight_penalty_col']] == 1][CONFIG['cost']].sum()
        cls.nonfreight_budget = cls.toy_data[cls.toy_data[CONFIG['freight_penalty_col']] == 0][CONFIG['cost']].sum()
        cls.all_budget = cls.toy_data[CONFIG['cost']].sum()

    def verify_result(self, assignments: pd.Series, budget: float, metro_range: tuple=None, freight_range: tuple=None):
        self.assertLessEqual(assignments.nunique(), 2)
        self.assertSetEqual(set(assignments.astype(int).unique()) - {1, 0}, set(), msg='Chosen flags are not binary!')
        # test budget
        chosen = self.toy_data[assignments == 1]
        self.assertLessEqual(chosen[CONFIG['cost']].sum(), budget, msg='Chosen projects overshoots budget')
        if chosen[CONFIG['cost']].sum() > 0 and (metro_range is not None):
            # test metro split
            met_reg_split = chosen[chosen[CONFIG['metro_penalty_col']] == 1][CONFIG['cost']].sum() / chosen[CONFIG['cost']].sum()
            self.assertLessEqual(metro_range[0], met_reg_split, msg='Metro/Regional Split not respected!')
            self.assertLessEqual(met_reg_split, metro_range[1], msg='Metro/Regional Split not respected!')
        if chosen[CONFIG['cost']].sum() > 0 and (freight_range is not None):
            # test metro split
            freight_split = chosen[chosen[CONFIG['freight_penalty_col']] == 1][CONFIG['cost']].sum() / chosen[CONFIG['cost']].sum()
            self.assertLessEqual(freight_range[0], freight_split, msg='Freight/Non-Freight Split not respected!')
            self.assertLessEqual(freight_split, freight_range[1], msg='Freight/Non-Freight Split not respected!')
    
    def test_no_budget_no_split_restriction(self):
        for name, method in PROBLEM_FORMS.items():
            with self.subTest(msg=f'Formulation {name}.'):
                solver, var_dict, _ = method(self.toy_data, penalties=dict(), budget=0, metro_range=(0, 1)) 
                result = make_solve_problem(solver, var_dict)
                self.verify_result(result, budget=0, metro_range=(0, 1))
                chosen = self.toy_data[result == 1]
                self.assertEqual(len(chosen), 0)

    def test_all_budget_no_split_restriction(self):
        for name, method in PROBLEM_FORMS.items():
            with self.subTest(msg=f'Formulation {name}.'):
                solver, var_dict, _ = method(self.toy_data, penalties=dict(), budget=self.all_budget, metro_range=(0, 1)) 
                result = make_solve_problem(solver, var_dict)
                self.verify_result(result, budget=self.all_budget, metro_range=(0, 1))
                chosen = self.toy_data[result == 1]
                self.assertEqual(len(chosen), len(result))

    def test_all_budget_metro_only(self):
        for name, method in PROBLEM_FORMS.items():
            with self.subTest(msg=f'Formulation {name}.'):
                solver, var_dict, _ = method(self.toy_data, penalties={CONFIG['metro_penalty_col']: -100}, budget=self.all_budget, metro_range=(1, 1)) 
                result = make_solve_problem(solver, var_dict)
                self.verify_result(result, budget=self.all_budget, metro_range=(1, 1))
                chosen = self.toy_data[result == 1]
                self.assertSetEqual(set(chosen[CONFIG['metro_penalty_col']]), {1})
                self.assertSetEqual(set(self.toy_data[result == 0][CONFIG['metro_penalty_col']]), {0})
                self.assertEqual(chosen[CONFIG['objective_col_with_treatment']].sum(), self.toy_data[self.toy_data[CONFIG['metro_penalty_col']] == 1][CONFIG['objective_col_with_treatment']].sum())

    def test_all_budget_reg_only(self):
        for name, method in PROBLEM_FORMS.items():
            with self.subTest(msg=f'Formulation {name}.'):
                solver, var_dict, _ = method(self.toy_data, penalties={CONFIG['metro_penalty_col']: 100}, budget=self.all_budget, metro_range=(0, 0)) 
                result = make_solve_problem(solver, var_dict)
                self.verify_result(result, budget=self.all_budget, metro_range=(0, 0))
                chosen = self.toy_data[result == 1]
                self.assertSetEqual(set(chosen[CONFIG['metro_penalty_col']]), {0})
                self.assertSetEqual(set(self.toy_data[result == 0][CONFIG['metro_penalty_col']]), {1})
                self.assertEqual(chosen[CONFIG['objective_col_with_treatment']].sum(), self.toy_data[self.toy_data[CONFIG['metro_penalty_col']] == 0][CONFIG['objective_col_with_treatment']].sum())

    def test_percentage_by_cost(self):
        toy_data = pd.DataFrame({
            CONFIG['metro_penalty_col']:   [1, 1, 0, 0],
            CONFIG['freight_penalty_col']: [1, 0, 1, 0],
            CONFIG['cost']: [100.0, 200.0, 100.0, 100.0],
            CONFIG['objective_col_with_treatment']: [100.0, 100.0, 500.0, 100.0],
            CONFIG['objective_col_without_treatment']: [0.0, 0.0, 0.0, 0.0],
            CONFIG['los_after_with_treatment']: [100, 100, 100, 100],
            CONFIG['los_after_without_treatment']: [0, 0, 0, 0],
        }) # if percentage constraint is by count, best solution is 0, 1, 3 for 300. If percentage is by cost, best solution is all for 800 
        for name, method in PROBLEM_FORMS.items():
            if 'penalties' in name:
                continue
            with self.subTest(msg=f'Formulation {name}.'):
                solver, var_dict, _ = method(toy_data, penalties={CONFIG['metro_penalty_col']: 100}, budget=self.all_budget, metro_range=(0.4, 0.7), freight_range=(0.3, 0.4)) 
                result = make_solve_problem(solver, var_dict)
                self.assertEqual(result.sum(), 4)
                val_dict = get_objective_value(toy_data, result)
                self.assertEqual(val_dict['metro_perc'], 0.6)
                self.assertEqual(val_dict['freight_perc'], 0.4)
                self.assertEqual(val_dict['dLoS'], 800)
                self.assertEqual(val_dict['Absolute_LoS'], 400)

if __name__ == '__main__':
    unittest.main()