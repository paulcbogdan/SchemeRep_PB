from scipy import stats

from atlas_utils import get_atlas
from modularity import get_binary_matrix
from old.networks import load_FC_for_Lifu
from old_Apr6.ttest_mat import get_stats_graphs
from old.plot_gen import plot_connectivity
from utils import pickle_wrap
import numpy as np


def do_between_fp(alpha_thresh=.0001):
    kwargs = {'fp': 'con3_fMRI',
              'split': False,
              'key': 'true',
              'key_vals': (True, False)
              }
    sn_inc_conn0, sn_conn0, age2idxs, sn_inc_activity0 = \
        pickle_wrap(load_FC_for_Lifu, None, kwargs=kwargs, easy_override=False, verbose=1, cache_dir='../../cache')

    kwargs = {'fp': 'vis3_fMRI',
              'split': False,
              'key': 'true',
              'key_vals': (True, False)
              }
    sn_inc_conn1, sn_conn1, age2idxs1, sn_inc_activity1 = \
        pickle_wrap(load_FC_for_Lifu, None, kwargs=kwargs, easy_override=False, verbose=1, cache_dir='../../cache')
    M_graph, SD_graph, SE_graph, N_graph, t_graph, p_graph, z_graph = \
        get_stats_graphs(sn_conn0, sn_conn1)
    p_graph = np.min([p_graph, 1 - p_graph], axis=0)
    z_graph[p_graph > alpha_thresh] = np.nan

    atlas = get_atlas(schaefer=False)

    # print(sn_inc_activity1.shape)
    # quit()
    print('test')
    ROIs_l = atlas['ROIs']
    for i, ROI in enumerate(ROIs_l):
        a0 = np.nanmean(sn_inc_activity0[:, 0, i, :], axis=-1)
        a1 = np.nanmean(sn_inc_activity1[:, 0, i, :], axis=-1)
        nans = np.isnan(a0)
        a0 = a0[~nans]
        a1 = a1[~nans]
        t, p = stats.ttest_rel(a0, a1, nan_policy='omit')
        if abs(t) > 3:
            print(f'{ROI}, {t=:.3f}')

    quit()

    plot_connectivity(z_graph, atlas['ticks'], atlas['tick_labels'], atlas['tick_lows'],
                      title=f'conceptual vs. visual | mean connectivity (YA + OA)', no_avg=True, cbar_label='t-value')

    num_signif = np.sum(p_graph < alpha_thresh) // 2
    num_correction = .05 / alpha_thresh
    num_ROIs = sn_conn0.shape[2]
    num_edges = num_ROIs * (num_ROIs - 1) / 2
    expected_FP = num_edges * alpha_thresh * 2
    alpha_thresh_z = stats.norm.ppf(alpha_thresh)
    print(f'{alpha_thresh=} (z = {alpha_thresh_z:.2f}), {num_correction=} | '
          f'{num_edges=}, {expected_FP=:.2f}')
    print(f'\tNumber of significant edges: {num_signif=}')
    quit()


def prep_conn_M():
    kwargs = {'fp': 'obj4_fMRI',
              'split': False,
              'key': 'true',
              'key_vals': (True, False)
              }
    sn_inc_conn0, sn_conn0, age2idxs, sn_inc_activity0 = \
        pickle_wrap(load_FC_for_Lifu, None, kwargs=kwargs, easy_override=False, verbose=1, cache_dir='../cache')
    conn = np.nanmean(sn_conn0, axis=0)
    _, mat = get_binary_matrix(conn)
    out = ''
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if j < i:
                continue
            if mat[i, j] == 1:
                out += f'{i+1}\t{j+1}\n'
    print(out)
    with open('networks.txt', 'w') as f:
        f.write(out)


if __name__ == '__main__':
    # do_between_fp()
    prep_conn_M()
