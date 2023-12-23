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
    # p_graph = np.min([p_graph, 1 - p_graph], axis=0)
    return M_graph, SD_graph, SE_graph, N_graph, t_graph, p_graph, z_graph


def perm_test(sn_inc_conn, age2idxs, alpha_thresh):
    shuffler = np.array([0, 1])
    biggests = []
    for nsim in range(1000):
        rand_splitter = np.random.choice(list(range(sn_inc_conn.shape[0])),
                                         size=sn_inc_conn.shape[0] // 2,
                                         replace=False)
        # print(rand_splitter)
        # quit()
        for i in range(sn_inc_conn.shape[0]):
            # if i in rand_splitter:
            #     shuffler = np.array([0, 1])
            # else:
            #     shuffler = np.array([1, 0])
            np.random.shuffle(shuffler)
            sn_inc_conn[i, :, :, :] = sn_inc_conn[i, shuffler, :, :]

        M1_graph, SD1_graph, SE1_graph, N1_graph, t1_graph, p1_graph, z1_graph = \
            get_stats_graphs(sn_inc_conn[age2idxs[1], 0, :, :],
                             sn_inc_conn[age2idxs[1], 1, :, :])
        M2_graph, SD2_graph, SE2_graph, N2_graph, t2_graph, p2_graph, z2_graph = \
            get_stats_graphs(sn_inc_conn[age2idxs[2], 0, :, :],
                             sn_inc_conn[age2idxs[2], 1, :, :])

        SD12 = ((SD1_graph ** 2)*(N1_graph - 1) +
                (SD2_graph ** 2)*(N2_graph - 1)) /\
                (N1_graph + N2_graph - 2)
        SE12 = np.sqrt(SD12 * (1/N1_graph + 1/N2_graph))
        t12_graph = (M1_graph - M2_graph) / SE12
        p12_graph = 1 - stats.t.cdf(t12_graph, N1_graph + N2_graph - 2)
        z12_graph = -stats.norm.ppf(p12_graph)
        p12_graph = np.min([p12_graph, 1 - p12_graph], axis=0)
        components, biggest_size = get_NBS_clusters(p12_graph, alpha_thresh)
        print(f'{nsim}: {biggest_size=}')
        biggests.append(biggest_size)
        for cutoff in [0.5, 0.1, 0.05, 0.01]:
            size = np.quantile(biggests, 1 - cutoff)
            print(f'{cutoff=}: {size=}')

        num_correction = .05 / alpha_thresh
        num_ROIs = sn_inc_conn.shape[2]
        num_edges = num_ROIs * (num_ROIs - 1) / 2
        expected_FP = num_edges * alpha_thresh * 2
        print(f'{alpha_thresh=}, {num_correction=} | '
              f'{num_edges=}, {expected_FP=:.2f}')
        num_signif = np.sum(p12_graph < alpha_thresh) // 2
        print(f'\tNumber of significant edges: {num_signif=}')
        print('------')
    quit()

def do_NBS(threshold=0.9, min_cluster_size=50, alpha_thresh=.001):
    fp = 'obj4_fMRI'
    kwargs = {'fp': fp,
              'split': False,
              'key': 'inc',
              'key_vals': (1, 3)
              }
    kwargs = {'fp': fp, 'split': False,
              'key': 'vis_hit',
              'key_vals': (False, True)}
    sn_inc_conn, sn_conn, age2idxs, sn_inc_activity = pickle_wrap(None,
                                                                  load_FC_for_Lifu, kwargs=kwargs,
                                                                  verbose=1, easy_override=False,
                                                                  cache_dir='../cache')
    cnt_trials0 = np.sum(~np.isnan(sn_inc_activity[:, 0, 0, :]))
    cnt_trials1 = np.sum(~np.isnan(sn_inc_activity[:, 1, 0, :]))
    ratio = cnt_trials0 / cnt_trials1
    # print(f'{cnt_trials0=}, {cnt_trials1=} | {ratio=:.3f}')
    # quit()

    # perm_test(sn_inc_conn, age2idxs, alpha_thresh)
    # quit()

    # quit()

    # print(f'{sn_inc_conn[:, :, 232, 0]=}')
    # quit()
    dif_graph = sn_inc_conn[age2idxs[1], 0, :, :] - \
                sn_inc_conn[age2idxs[1], 1, :, :]
    # print(f'{dif_graph[:, 232, 0]=}')



    #


    M1_graph, SD1_graph, SE1_graph, N1_graph, t1_graph, p1_graph, z1_graph = \
        get_stats_graphs(sn_inc_conn[age2idxs[1], 0, :, :],
                         sn_inc_conn[age2idxs[1], 1, :, :])
    M2_graph, SD2_graph, SE2_graph, N2_graph, t2_graph, p2_graph, z2_graph = \
        get_stats_graphs(sn_inc_conn[age2idxs[2], 0, :, :],
                         sn_inc_conn[age2idxs[2], 1, :, :])

    M12_graph, SD12_graph, SE12_graph, N12_graph, t12_graph, p12_graph, z12_graph = \
        get_stats_graphs(sn_inc_conn[:, 0, :, :],
                         sn_inc_conn[:, 1, :, :])

    # SD12 = ((SD1_graph ** 2) * (N1_graph - 1) +
    #         (SD2_graph ** 2) * (N2_graph - 1)) / \
    #        (N1_graph + N2_graph - 2)
    # SE12 = np.sqrt(SD12 * (1 / N1_graph + 1 / N2_graph))
    # t12_graph = (M1_graph - M2_graph) / SE12
    # p12_graph = 1 - stats.t.cdf(t12_graph, N1_graph + N2_graph - 2)
    # z12_graph = -stats.norm.ppf(p12_graph)
    # # p12_graph = p12_graph_
    p12_graph = np.min([p12_graph, 1 - p12_graph], axis=0)
    p1_graph = np.min([p1_graph, 1 - p1_graph], axis=0)
    p2_graph = np.min([p2_graph, 1 - p2_graph], axis=0)


    # p12_graph[p1_graph > .01] = 0.999

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
    # quit()ZZZZ

    num_correction = .05 / alpha_thresh
    num_ROIs = sn_inc_conn.shape[2]
    num_edges = num_ROIs * (num_ROIs - 1) / 2
    expected_FP = num_edges * alpha_thresh * 2
    print(f'{alpha_thresh=}, {num_correction=} | '
          f'{num_edges=}, {expected_FP=:.2f}')
    print(f'\tNumber of significant edges: {num_signif=}')
    # quit()

    components, biggest_size = get_NBS_clusters(p12_graph, alpha_thresh)
    coords = get_BNA_coords()

    dir_out = 'nichord_plots/NBS'
    for i, edges in enumerate(components):
        print(f'num edges: {len(edges)} | {edges=}')
        if len(edges) < min_cluster_size:
            continue
        edge_weights = [z12_graph[x][y] for (x, y) in edges]
        # for edge, w in zip(edges, edge_weights):
        #     print(f'{edge=}, {w=:.3f} | '
        #           f'M1={M1_graph[edge]:.3f}, M2={M2_graph[edge]:.3f}')
        fn = f'age_x_vis_NBS_a{alpha_thresh}_c{i}.png'
        title = f'NBS group: {i}'
        title = f'Main effect of congruency (YA + OA), primary threshold: p < {alpha_thresh}'
        title = f'Main effect of visual memory (YA + OA), primary threshold: p < {alpha_thresh}'

        plot_nichord(coords, fn, title, dir_out=dir_out,
                     edges=edges, edge_weights=edge_weights)
        print('plotted')


if __name__ == '__main__':
    do_NBS()




