
import os
from pathlib import Path

from sklearn.svm import SVC

from old.network_funcs import load_FC_for_Lifu

os.chdir(r'E:\PycharmProjects_E\SchemeRep')
from connsearch.permute import Permutation_Manager
from connsearch.components import get_none_components

from utils import PICKLE_CACHE, pickle_wrap
import numpy as np

if __name__ == '__main__':
    kwargs = {'fp': 'obj7_fMRI',
              'split': False,
              'key': 'inc',
              'key_vals': (1, 3),
              'combine_regions': False,
              }
    sn_inc_conn, sn_conn, age2idxs, sn_inc_activity = \
        pickle_wrap(load_FC_for_Lifu, None, kwargs=kwargs,
                    easy_override=False, verbose=1, cache_dir='cache')
    sn_inc_conn[np.isnan(sn_inc_conn)] = 0
    X_GLOBAL = sn_inc_conn[1:, None]
    Y_GLOBAL = [[0, 1] for _ in range(X_GLOBAL.shape[0])]
    Y_GLOBAL = np.array(Y_GLOBAL)[:, None]
    COORDS = None
    # print(sn_inc_conn.shape)
    # quit()

    # X_GLOBAL, Y_GLOBAL, COORDS = generate_dataset()


    N_PERM = 1000  # Number of shuffled datasets to analyze
    N_SPLITS = 2  # Number of splits for cross-validation
    N_REPEATS = 25  # Number of repeats for cross-validation
    COMP_SIZE = None # Component size
    CACHE_DIR = f'cache/permutation_saves'
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    PERMUTATION_SCHEME = 'within_session'  # Specifies how data will be shuffled
    COMPONENT_FUNC = get_none_components  # Use connectivity sets as components

    PM = Permutation_Manager(X_GLOBAL, Y_GLOBAL, COORDS, SVC(kernel='rbf'),
                             comp_size=COMP_SIZE,
                             n_folds=N_SPLITS,
                             component_func=COMPONENT_FUNC,
                             wu_analysis=True,
                             n_repeats=N_REPEATS,
                             n_perm=N_PERM,
                             perm_strategy=PERMUTATION_SCHEME,
                             cache_dir=CACHE_DIR)
    PM.run_permutations()



















