import os
from collections import defaultdict

import numpy as np

from atlas_utils import get_atlas
from ttest_mat import get_stats_graphs
from old.network_funcs import load_FC_for_Lifu
from plot_conn_nice import HC_t
from utils import pickle_wrap
from vendor_partitioning import get_vendor_partitions_, get_vendor_partitions, get_anat_vendor_partitions

os.chdir('E:\PycharmProjects_E\SchemeRep')

def plot_quadrants(anat=True):
    kwargs = {'fp': 'obj7_fMRI',
              'key': 'inc',
              'atlas_name': 'BNA',
              'key_vals': (1, 3),
              }
    sn_inc_conn, sn_conn, age2idxs, sn_inc_activity = \
        pickle_wrap(None, load_FC_for_Lifu, kwargs=kwargs, verbose=1,
                    easy_override=False, cache_dir='cache')
    age2idxs['healthy'] = age2idxs[1] + age2idxs[2]

    # if anat:
    # else:
    p_dorsal, p_ventral, p_d_ant, p_d_pos, p_v_ant, p_v_pos, matrix_mask = \
        get_vendor_partitions(age='healthy', flip=True, anat=anat)

    quads = [p_d_ant, p_d_pos, p_v_ant, p_v_pos]
    anat_str = ' (anat)' if anat else ''
    names = [f'Seed: PFC{anat_str}',
             f'Seed: Parietal{anat_str}',
             f'Seed: ATL{anat_str}',
             f'Seed: Occipital{anat_str}']
    atlas = get_atlas()
    for p, name in zip(quads, names):
        accs_faux = {name: defaultdict(lambda: None)}
        HC_t(age2idxs, sn_inc_activity[:, :, p, :], sn_inc_activity,
             kwargs, name, accs_faux, atlas)


if __name__ == '__main__':
    plot_quadrants()