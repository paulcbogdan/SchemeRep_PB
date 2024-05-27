
import os
from pathlib import Path

from connsearch import ConnSearcher
from connsearch.report import prepare_components_table
from sklearn.svm import SVC

from atlas_utils import get_atlas
from old.network_funcs import load_FC_for_Lifu

from connsearch.permute import Permutation_Manager
from connsearch.components import get_none_components, get_components

from utils import PICKLE_CACHE, pickle_wrap
import numpy as np
import matplotlib.pyplot as plt

def do_actual():
    dir_results = r'cache/cs_results'
    coords = get_atlas()['coords']
    clf = SVC(kernel='linear')
    components = get_components(no_components=True)
    CS = ConnSearcher(X_GLOBAL, Y_GLOBAL, coords, components, dir_results,
                      n_splits=2, n_repeats=100, clf=clf,
                      wu_analysis=True)
    CS.do_group_level_analysis(acc_thresh=.6)

    fn_csv = f'ConnSearch_Group.csv'
    dir_table = Path(dir_results).parent
    fp_csv = os.path.join(dir_table, fn_csv)
    prepare_components_table(fp_csv, dir_results)

if __name__ == '__main__':
    kwargs = {'fp': 'obj7_fMRI',
              'split': False,
              'key': 'inc',
              'key_vals': (1, 2, 3),
              'combine_regions': False,
              }
    sn_inc_conn, sn_conn, age2idxs, sn_inc_activity = \
        pickle_wrap(load_FC_for_Lifu, None, kwargs=kwargs,
                    easy_override=False, verbose=1, cache_dir='cache')

    # sn_inc_conn = sn_inc_conn[:, [0, 2], :, :]

    sn_inc_conn[np.isnan(sn_inc_conn)] = 0
    X_GLOBAL = sn_inc_conn[1:, None]
    print(X_GLOBAL.shape)
    # print(X_GLOBAL.shape)
    # print(X_GLOBAL.mean(axis=2).shape)
    # quit()
    # X_GLOBAL -= X_GLOBAL.mean(axis=2)[:, :, None]
    X_GLOBAL -= X_GLOBAL[:, :, [1]]
    X_GLOBAL = X_GLOBAL[:, :, [0, 2]]
    # print(X_GLOBAL.shape)
    # quit()


    # print(X_GLOBAL[:, :, :, 210, 200])
    # quit()

    Y_GLOBAL = [[0, 1] for _ in range(X_GLOBAL.shape[0])]
    Y_GLOBAL = np.array(Y_GLOBAL)[:, None]
    do_actual()

    COORDS = None
    N_PERM = 1000  # Number of shuffled datasets to analyze
    N_SPLITS = 2  # Number of splits for cross-validation
    N_REPEATS = 100  # Number of repeats for cross-validation
    COMP_SIZE = None # Component size
    CACHE_DIR = f'cache/permutation_saves'
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    PERMUTATION_SCHEME = 'within_session'  # Specifies how data will be shuffled
    COMPONENT_FUNC = get_none_components  # Use connectivity sets as components

    PM = Permutation_Manager(X_GLOBAL, Y_GLOBAL, COORDS, SVC(kernel='linear'),
                             comp_size=COMP_SIZE,
                             n_folds=N_SPLITS,
                             component_func=COMPONENT_FUNC,
                             wu_analysis=True,
                             n_repeats=N_REPEATS,
                             n_perm=N_PERM,
                             perm_strategy=PERMUTATION_SCHEME,
                             cache_dir=CACHE_DIR)
    PM.run_permutations()



















