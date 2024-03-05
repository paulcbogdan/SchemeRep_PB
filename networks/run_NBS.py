import os

from ttest_mat import get_stats_graphs

os.chdir('E:\PycharmProjects_E\SchemeRep')

from scipy import stats

from atlas_utils import get_atlas
from old.modularity import get_main_partitions, get_partition_matrix
from old.network_funcs import load_FC_for_Lifu
from old.plot_gen import plot_connectivity
from utils import pickle_wrap
from old.NBS import get_NBS_clusters
import numpy as np
from pprint import pprint
from nilearn import plotting
import matplotlib.pyplot as plt

def edges2adj(edges, z, ar_size=246):
    adj = np.zeros((ar_size, ar_size))
    for (i, j) in edges:
        adj[i, j] = z[i, j]
        adj[j, i] = z[j, i]
    return adj

def drop_w_few_edges(adj, alpha, limit=2):
    bad_is = []
    for i, row in enumerate(adj):
        if np.sum(row < alpha) < limit:
            bad_is.append(i)
    for i in bad_is:
        adj[i, :] = .5
        adj[:, i] = .5
    return adj

def run_NBS(fp='obj7_fMRI', combine_regions=False, alpha=.0025):
    kwargs = {'fp': fp,
              'split': False,
              'key': 'inc',
              'key_vals': (1, 2, 3),
              'combine_regions': combine_regions,
              }
    sn_inc_conn, sn_conn, age2idxs, sn_inc_activity = \
        pickle_wrap(load_FC_for_Lifu, None, kwargs=kwargs,
                    easy_override=False, verbose=1, cache_dir='cache')

    _, _, _, _, _, p_graph, z_graph = \
        get_stats_graphs(sn_inc_conn[:, 0, :, :],
                         sn_inc_conn[:, 2, :, :])

    atlas = get_atlas()
    bad_rois = {'Amyg', 'Hipp', 'Str', 'Tha'}
    bad_j = [j for j, roi in enumerate(atlas['ROI_regions'])
             if roi in bad_rois]
    for j in bad_j:
        z_graph[:, j] = 0
        z_graph[j, :] = 0
        p_graph[:, j] = .5
        p_graph[j, :] = .5

    # p2_graph = 1 - p2_graph
    # print(f'{z_graph.shape=}')
    # z2_graph = -z2_graph
    p_graph = stats.norm.cdf(z_graph)
    # print(np.nanmin(p2_graph))
    # quit()
    p_graph = drop_w_few_edges(p_graph, alpha)
    components, biggest_size = get_NBS_clusters(p_graph, alpha)
    # pprint(components)
    coords = atlas['coords']
    for i, c in enumerate(components):
        print(f'Component {i} has {len(c)} edges.')
        # continue
        if len(c) > 10:
            adj = edges2adj(c, z_graph)
            plotting.plot_connectome(adj, coords, #edge_threshold='99.9%',
                                     node_size=10, colorbar=True)
            plt.show()





if __name__ == '__main__':
    run_NBS()






