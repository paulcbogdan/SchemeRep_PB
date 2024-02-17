import numpy as np

from ttest_mat import get_stats_graphs
from old.networks import load_FC_for_Lifu
from modularity import get_main_partitions, get_partition_matrix
from utils import pickle_wrap
import scipy.stats as stats

def get_acc(df_sn):
    return df_sn['con_hit'].mean()


def brain_x_bhv():
    fp = 'obj7_fMRI'
    kwargs = {'fp': fp,
              'split': False,
              'key': 'inc',
              'key_vals': (1, 3),
              'get_df_sn': True
              }

    sn_inc_conn, sn_conn, age2idxs, sn_inc_activity, df_sns = \
        pickle_wrap(None, load_FC_for_Lifu, kwargs=kwargs, verbose=1,
                    easy_override=False, cache_dir='cache')

    M1_graph, SD1_graph, SE1_graph, N1_graph, t2_graph, p1_graph, z2_graph = \
        get_stats_graphs(sn_inc_conn[age2idxs[1], 0, :, :],
                         sn_inc_conn[age2idxs[1], 1, :, :])

    age_df_sns = [df_sns[i] for i in age2idxs[1]]
    accs = list(map(get_acc, age_df_sns))

    flip_t = False
    z2_graph = -z2_graph if flip_t else z2_graph
    partitions, matrix_mask = get_main_partitions(z2_graph, coords=None,
                                                  plot=True,
                        threshold=.95, fn_str='', overlapping=False,
                        dir_out=f'OA2_ttest_modules_thr{.95}_flip{flip_t}.png')

    difs = sn_inc_conn[age2idxs[1], 0, :, :] - sn_inc_conn[age2idxs[1], 1, :, :]
    for i, p0 in enumerate(partitions):
        pmat = get_partition_matrix(difs, p0)
        M_conn = np.nanmean(pmat, axis=(-2, -1))
        r, p = stats.spearmanr(M_conn, accs)
        print(f'p{i}: {r=:.2f}, {p=:.3f}')

if __name__ == '__main__':
    brain_x_bhv()

