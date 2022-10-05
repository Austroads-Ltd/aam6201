import numpy as np
import unittest
import time

from src.inner_strategy import early_stop, optimal_solve
from src.forms.ip_form import make_problem_penalties
from src.util import load_data, make_solve_problem, var_dict_to_series
from src.config import CONFIG

class TestStrategy(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.toy_data = load_data()
        cls.metro_budget = cls.toy_data[cls.toy_data[CONFIG['metro_penalty_col']] == 1][CONFIG['cost']].sum()
        cls.reg_budget = cls.toy_data[cls.toy_data[CONFIG['metro_penalty_col']] == 0][CONFIG['cost']].sum()
        cls.freight_budget = cls.toy_data[cls.toy_data[CONFIG['freight_penalty_col']] == 1][CONFIG['cost']].sum()
        cls.nonfreight_budget = cls.toy_data[cls.toy_data[CONFIG['freight_penalty_col']] == 0][CONFIG['cost']].sum()
        cls.all_budget = cls.toy_data[CONFIG['cost']].sum()

    def test_optimal(self):
        budget = 50000000 
        solver_1, var_dict_1, _ = make_problem_penalties(self.toy_data, penalties=dict(), budget=budget, metro_range=(0, 1)) 
        (status, solver_2, var_dict_2, _), _ = optimal_solve(self.toy_data, make_problem_penalties, penalties=dict(), budget=budget, metro_range=(0, 1)) 
        result_1 = make_solve_problem(solver_1, var_dict_1)
        self.assertTrue(status == 0, "Optimal strategy not optimal!")
        self.assertTrue(np.all(result_1 == var_dict_to_series(var_dict_2)), "Optimal strategy not equal to normal result!")

    def test_early_stop_optimal_capable(self):
        budget = 50000000
        solver_1, var_dict_1, _ = make_problem_penalties(self.toy_data, penalties=dict(), budget=budget, metro_range=(0, 1)) 
        (status, solver_2, var_dict_2, _), _ = early_stop(self.toy_data, make_problem_penalties, timeout=1000, patience=3, penalties=dict(), budget=budget, metro_range=(0, 1)) 
        result_1 = make_solve_problem(solver_1, var_dict_1)
        self.assertTrue(status == 0, "Early stop stops too early!")
        self.assertTrue(np.all(result_1 == var_dict_to_series(var_dict_2)), "Early stop not producing optimal result when it can!")

    def test_early_stop_nonoptimal(self):
        budget = 200000000
        start = time.time()
        (status, _, _, _), _ = early_stop(self.toy_data, make_problem_penalties, timeout=1000, patience=5, penalties=dict({CONFIG['metro_penalty_col']: -10}), budget=budget, metro_range=(0, 1)) 
        self.assertFalse(status == 0, "Early stop optimal flag incorrect")
        self.assertLess(time.time() - start, 8, 'Early stop taking too much time!')


if __name__ == "__main__":
    unittest.main()