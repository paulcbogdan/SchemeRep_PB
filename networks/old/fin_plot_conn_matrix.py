import numpy as np
from scipy import stats as stats

from atlas_utils import get_atlas
from ttest_mat import get_stats_graphs, get_2sample_graph
from old.network_funcs import load_FC_for_Lifu
from old.plot_gen import plot_connectivity
from utils import pickle_wrap


def plot_conn_matrix(combine_regions=True):
    kwargs = {'fp': 'obj7_fMRI',
              'key': 'inc',
              'key_vals': (1, 3),
              'combine_regions': combine_regions
              }
    sn_inc_conn, sn_conn, age2idxs, sn_inc_activity = \
        pickle_wrap(load_FC_for_Lifu, None, kwargs=kwargs, easy_override=False, verbose=1, cache_dir='../cache')

    _, _, _, _, _, p_both, z_both = \
        get_stats_graphs(sn_inc_conn[:, 0, :, :],
                         sn_inc_conn[:, 1, :, :])
    M1_graph, SD1_graph, SE1_graph, N1_graph, t1_graph, p_YA, z_YA = \
        get_stats_graphs(sn_inc_conn[age2idxs[1], 0, :, :],
                         sn_inc_conn[age2idxs[1], 1, :, :])

    M2_graph, SD2_graph, SE2_graph, N2_graph, t2_graph, p_OA, z_OA = \
        get_stats_graphs(sn_inc_conn[age2idxs[2], 0, :, :],
                         sn_inc_conn[age2idxs[2], 1, :, :])

    dif2_graph = sn_inc_conn[age2idxs[2], 0, :, :] - \
                 sn_inc_conn[age2idxs[2], 1, :, :]
    dif1_graph = sn_inc_conn[age2idxs[1], 0, :, :] - \
                 sn_inc_conn[age2idxs[1], 1, :, :]
    z_intr, p_intr = get_2sample_graph(dif2_graph, dif1_graph)
    p_graph = p_intr

    alpha_thresh = .01
    num_signif = np.sum(p_graph < alpha_thresh) // 2
    num_correction = .05 / alpha_thresh
    num_ROIs = sn_inc_conn.shape[2]
    num_edges = num_ROIs * (num_ROIs - 1) / 2
    expected_FP = num_edges * alpha_thresh
    alpha_thresh_z = stats.norm.ppf(alpha_thresh)
    print(f'{alpha_thresh=} (z = {alpha_thresh_z:.2f}), {num_correction=} | '
          f'{num_edges=}, {expected_FP=:.2f}')
    print(f'\tNumber of significant edges: {num_signif=}')

    atlas = get_atlas(combine_regions=kwargs['combine_regions'])
    fp2name = {'obj7_fMRI': 'Object encoding',
               'con3_fMRI': 'Conceptual retrieval',
               'vis3_fMRI': 'Visual retrieval'}
    z_OA[p_OA > .1] = np.nan
    z_YA[p_YA > .1] = np.nan
    z_both[p_both > .1] = np.nan
    z_intr[p_intr > .1] = np.nan

    plot_kwargs = {'ticks': atlas['ticks'],
                   'tick_labels': atlas['tick_labels'],
                   'tick_lows': atlas['tick_lows'],
                   'no_avg': True,

                   'vmin': -3.25, 'vmax': 3.25,
                   'cbar_label': 'z-value',
                   'tile': .001,
                   'tick_low': '(Con)',
                   'tick_high': '(Inc)'}
    plot_connectivity(z_OA,
                      title=f'{fp2name[kwargs["fp"]]}\n'
                            f'Congruency effect (OA)',
                      **plot_kwargs)
    plot_connectivity(z_YA,
                      title=f'{fp2name[kwargs["fp"]]}\n'
                            f'Congruency effect (YA)',
                      **plot_kwargs)
    plot_connectivity(z_both,
                     title=f'{fp2name[kwargs["fp"]]}\n'
                           f'Congruency effect (YA + OA)',
                     **plot_kwargs)
    plot_kwargs['tick_low'] = f'\n(Inc higher in OA)'
    plot_kwargs['tick_high'] = f'\n(Con higher in OA)'
    plot_connectivity(z_intr,
                     title=f'{fp2name[kwargs["fp"]]}\n'
                           f'Congruency x Age interaction',
                     **plot_kwargs)

if __name__ == '__main__':
    plot_conn_matrix()