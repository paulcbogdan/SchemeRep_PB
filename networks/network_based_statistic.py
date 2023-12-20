from matplotlib import pyplot as plt
from nichord import get_idx_to_label
from nichord.combine import plot_and_combine
from scipy import stats

from modularity import get_BNA_coords, plot_nichord
from network_funcs import load_FC_for_Lifu
from utils import pickle_wrap
from NBS import get_NBS_clusters
import numpy as np

def get_stats_graphs(graph0, graph1):
    dif_graph = graph0 - graph1
    M_graph = np.nanmean(dif_graph, axis=0)
    SD_graph = np.nanstd(dif_graph, axis=0)
    N_graph = np.nansum(~np.isnan(dif_graph), axis=0)
    SE_graph = SD_graph / (N_graph ** 0.5)
    t_graph = M_graph / SE_graph
    p_graph = 1 - stats.t.cdf(t_graph, N_graph - 1)
    z_graph = stats.norm.ppf(p_graph)
    return M_graph, SD_graph, SE_graph, N_graph, t_graph, p_graph, z_graph

def do_NBS(threshold=0.9, min_cluster_size=10, alpha_thresh=.00005):
    fp = 'obj4_fMRI'
    kwargs = {'fp': fp,
              'split': False,
              'key': 'inc',
              'key_vals': (1, 3)
              }
    kwargs = {'fp': fp, 'split': False,
              'key': 'con_hit',
              'key_vals': (False, True)}
    sn_inc_conn, sn_conn, age2idxs, sn_inc_activity = pickle_wrap(None,
                                                                  load_FC_for_Lifu, kwargs=kwargs,
                                                                  verbose=1, easy_override=False,
                                                                  cache_dir='../cache')


    # quit()

    # print(f'{sn_inc_conn[:, :, 232, 0]=}')
    # quit()
    dif_graph = sn_inc_conn[age2idxs[1], 0, :, :] - \
                sn_inc_conn[age2idxs[1], 1, :, :]
    # print(f'{dif_graph[:, 232, 0]=}')

    M1_graph, SD1_graph, SE1_graph, N1_graph, t1_graph, p1_graph, z1_graph = \
        get_stats_graphs(sn_inc_conn[age2idxs[1], 0, :, :], 
                         sn_inc_conn[age2idxs[1], 1, :, :])
    M2_graph, SD2_graph, SE2_graph, N2_graph, t2_graph, p2_graph, z2_graph = \
        get_stats_graphs(sn_inc_conn[age2idxs[2], 0, :, :],
                         sn_inc_conn[age2idxs[2], 1, :, :])

    SE12 = np.sqrt(((SE1_graph ** 2)*(N1_graph - 1) +
                    (SE2_graph ** 2)*(N2_graph - 1)) /
                   (N1_graph + N2_graph - 2))
    t12_graph = (M1_graph - M2_graph) / SE12
    p12_graph = 1 - stats.t.cdf(t12_graph, N1_graph + N2_graph - 2)
    z12_graph = -stats.norm.ppf(p12_graph)
    p12_graph = np.min([p12_graph, 1 - p12_graph], axis=0)

    
    # p1_graph = np.min([p1_graph, 1 - p1_graph], axis=0)
    num_signif = np.sum(p12_graph < alpha_thresh) // 2
    signif_graph = p12_graph < alpha_thresh
    signif_z_graph = z12_graph.copy()
    signif_z_graph[~signif_graph] = np.nan
    fig, axs = plt.subplots(1, 2)
    plt.sca(axs[0])
    plt.imshow(z12_graph)
    plt.title('z-graph')
    plt.colorbar()
    plt.sca(axs[1])
    plt.imshow(signif_z_graph, interpolation='none', cmap='RdBu')
    plt.title('p-graph')
    plt.show()
    # quit()

    num_correction = .05 / alpha_thresh
    num_ROIs = sn_inc_conn.shape[2]
    num_edges = num_ROIs * (num_ROIs - 1) / 2
    expected_FP = num_edges * alpha_thresh
    print(f'{alpha_thresh=}, {num_correction=} | '
          f'{num_edges=}, {expected_FP=:.2f}')
    print(f'\tNumber of significant edges: {num_signif=}')

    components, biggest_size = get_NBS_clusters(p12_graph, alpha_thresh)
    coords = get_BNA_coords()

    dir_out = '../nichord_plots/NBS'
    for i, edges in enumerate(components):
        print(f'num edges: {len(edges)} | {edges=}')
        if len(edges) < min_cluster_size:
            continue
        edge_weights = [z12_graph[x][y] for (x, y) in edges]
        for edge, w in zip(edges, edge_weights):
            print(f'{edge=}, {w=:.3f} | '
                  f'M1={M1_graph[edge]:.3f}, M2={M2_graph[edge]:.3f}')
        fn = f'age_x_inc_NBS_a{alpha_thresh}_c{i}.png'
        title = f'NBS group: {i}'
        plot_nichord(coords, fn, title, dir_out=dir_out,
                     edges=edges, edge_weights=edge_weights)


if __name__ == '__main__':
    do_NBS()




