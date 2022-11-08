"""
Meta optimizer util
"""

import pandas as pd
import numpy as np
import contextlib
import pickle
import datetime
import joblib
import time

from src.inner_strategy import early_stop 
from src.forms.ip_form import make_problem_penalties
from src.util import load_data, get_objective_value, var_dict_to_series
from src.config import CONFIG
from itertools import product
from joblib import Parallel, delayed
from tqdm import tqdm    

from src import DATA_DIR
from pathlib import Path
from typing import List, Tuple


# TQDM Wrapper for parallelization
# https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution
@contextlib.contextmanager
def tqdm_joblib(tqdm_object: tqdm):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

class UnsolveableError(Exception):
    pass

def grid_to_output():
    # dataset
    df = load_data()
    assert set(df.index) == set(range(len(df)))

    # generate grid of penalties
    metro_pen = np.linspace(-100, 100, num=101)
    freight_pen = np.linspace(-100, 100, num=101)

    # to save time on pickling/unpickling the same data, write the dataset into a file and read from it
    df_pickle = pickle.dumps(df) 
    def make_solve_problem_wrapper(df_pickle, penalties: dict={}, **kwargs):
        df = pickle.loads(df_pickle)
        (status, _, var_dict, objective), verifications = early_stop(df, make_problem_penalties, patience=3, penalties=penalties, **kwargs)
        if (status != 0 and status != 1):
            raise UnsolveableError()

        relative_gap = abs(objective.BestBound() - objective.Value()) / objective.BestBound()
        return penalties, relative_gap, var_dict_to_series(var_dict), verifications
                

    start = time.time()
    results : List[Tuple[dict, float, pd.Series, list]] = []

    # for m_p, f_p in tqdm(product(metro_pen, freight_pen), total=len(metro_pen)*len(freight_pen)):
    #     pen_dict = {CONFIG['metro_penalty_col']: m_p, CONFIG['freight_penalty_col']: f_p}
    #     is_optimal, selection = early_stop(df, patience=3, penalties=penalties, budget=float(os.getenv("budget")))
    #     results.append((pen_dict, is_optimal, selection)) 

    print("Making job queue: ")
    jobs = []
    for m_p, f_p in tqdm(product(metro_pen, freight_pen), total=len(metro_pen)*len(freight_pen)):
        pen_dict = {CONFIG['metro_penalty_col']: m_p, CONFIG['freight_penalty_col']: f_p}
        jobs.append(delayed(make_solve_problem_wrapper)(df_pickle, budget=float(CONFIG['budget']), penalties=pen_dict))

    with tqdm_joblib(tqdm(desc='Inner Optimization', total=len(jobs))):
        results.extend(Parallel(n_jobs=CONFIG["njobs"], batch_size=10)(jobs))

    print("Finished in {}".format(datetime.timedelta(seconds=time.time()-start)))
    res_list = []
    for p_dict, gap, r, vers in results:
        val_dict = get_objective_value(df, r)
        val_dict.update({
            CONFIG['metro_penalty_col']: p_dict[CONFIG['metro_penalty_col']],
            CONFIG['freight_penalty_col']: p_dict[CONFIG['freight_penalty_col']],
            'budget': float(CONFIG['budget']),
            'selection': r.to_list(),
            'gap': gap ,
            'verification_result': vers[-1]
        })
        res_list.append(val_dict)

    result_df = pd.DataFrame(res_list)
    save_path = DATA_DIR / f'{CONFIG["output_filename"].replace(".csv", "")}.csv'
    if save_path.exists():
        existing_df = pd.read_csv(save_path)
        result_df = existing_df.append(result_df).reset_index(drop=True)
    result_df.to_csv(save_path, index=False)

if __name__ == "__main__":
    grid_to_output()
