from matplotlib import pyplot as plt
from nichord import get_idx_to_label
from nichord.combine import plot_and_combine
from scipy import stats

from atlas_utils import get_atlas
from modularity import get_BNA_coords, plot_nichord, get_main_partitions, get_partition_matrix
from network_funcs import load_FC_for_Lifu
from old.plot_gen import plot_connectivity
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

def plot_M():
    fp = 'obj4_fMRI'
    fp = 'con3_fMRI'
    kwargs = {'fp': fp,
              'split': False,
              'key': 'inc',
              'key_vals': (1, 3)
              }
    sn_inc_conn, sn_conn, age2idxs, sn_inc_activity = \
        pickle_wrap(None, load_FC_for_Lifu, kwargs=kwargs, verbose=1,
                    easy_override=False, cache_dir='cache')
    print(sn_inc_conn.shape)
    atlas = get_atlas(schaefer=False)
    fp2name = {'obj4_fMRI': 'obj. encoding',
               'con3_fMRI': 'conc. retrieval',
               'vis3_fMRI': 'vis. retrieval'}
    M_graph = np.nanmean(sn_inc_conn, axis=(0, 1))
    plot_connectivity(M_graph,
                      atlas['ticks'],
                      atlas['tick_labels'],
                      atlas['tick_lows'],
                      no_avg=True,
                      title=f'{fp2name[kwargs["fp"]]} | mean connectivity (YA + OA)',
                      # vmin=-4, vmax=4,
                      cbar_label='t-value')

def ANOVA_edges():
    fp = 'scn4_fMRI'
    kwargs = {'fp': fp,
              'split': False,
              'key': 'scn_cat',
              'key_vals': ('indoor', 'outdoor_developed', 'outdoor_nature')
              }
    fp = 'obj4_fMRI'
    kwargs = {'fp': fp,
              'split': False,
              'key': 'obj_cat',
              'key_vals': ('living_animal', 'dead_small', 'dead_large',
                           'living_plant', 'dead_medium')
              }
    # fp = 'obj4_fMRI'
    # kwargs = {'fp': fp,
    #           'split': False,
    #           'key': 'inc',
    #           'key_vals': (1, 3)
    #           }
    sn_inc_conn, sn_conn, age2idxs, sn_inc_activity = \
        pickle_wrap(None, load_FC_for_Lifu, kwargs=kwargs, verbose=1,
                    easy_override=False, cache_dir='cache')

    n_groups = sn_inc_conn.shape[1]
    print(f'{n_groups=}')
    n_sn = np.sum(~np.isnan(sn_inc_conn[:, 0, :, :]), axis=0)
    M_total = np.nanmean(sn_inc_conn, axis=(0, 1))
    M_cond = np.nanmean(sn_inc_conn, axis=0)
    SSM = np.nansum((M_cond - M_total) ** 2, axis=0) * n_sn
    M_subj = np.nanmean(sn_inc_conn, axis=1)
    SSW = np.nansum((sn_inc_conn - M_subj[:, None, :, :]) ** 2, axis=(0, 1))

    SSR = SSW - SSM
    MSM = SSM / (n_groups - 1)
    MSR = SSR / ((n_sn - 1) * (n_groups - 1))
    # print(f'{n_sn[0, 1]=}')
    F = MSM / MSR
    # print(f'{F[0, 1]=}')
    p = 1 - stats.f.cdf(F, n_groups - 1, (n_sn - 1) * (n_groups - 1))
    # print(f'{p[0, 1]=}')
    z = stats.norm.ppf(p)
    # z[p < .99] = np.nan

    # dif = sn_inc_conn[:, 0, :, :] - sn_inc_conn[:, 1, :, :]
    # M_dif = np.nanmean(dif, axis=0)
    # SE_dif = np.nanstd(dif, axis=0, ddof=1) / np.sqrt(n_sn)
    # t = abs(M_dif / SE_dif)
    # print(f'{t[0, 1]=}')
    # p = 1 - stats.t.cdf(t, n_sn - 1)
    # print(f'{p[0, 1]=}')

    alpha_thresh = .05
    num_signif = np.sum(p < alpha_thresh) // 2
    num_correction = .05 / alpha_thresh
    num_ROIs = sn_inc_conn.shape[2]
    num_edges = num_ROIs * (num_ROIs - 1) / 2
    expected_FP = num_edges * alpha_thresh
    alpha_thresh_z = stats.norm.ppf(alpha_thresh)
    median_F = np.nanmedian(F)
    print(f'{median_F=:.3f}')
    # plt.hist(F.flatten(), bins=100, range=(0, 10))
    # plt.show()
    # quit()
    print(f'{alpha_thresh=} (z = {alpha_thresh_z:.2f}), {num_correction=} | '
          f'{num_edges=}, {expected_FP=:.2f}')
    print(f'\tNumber of significant edges: {num_signif=}')
    # quit()

    # t = np.sqrt(F)
    # plt.title(f'{np.nanmedian(F)=:.3f}')
    # plt.hist(t.flatten())
    # plt.show()
    # # p = 1 - stats.f.cdf(F, n_groups - 1, (n_sn - 1) * (n_groups - 1))
    # # z = stats.norm.ppf(p)
    #

    # # print(f'{t_dif[0, 1]**2=}')
    # p = 1 - stats.t.cdf(t, n_sn - 1)
    # # print(f'paired t-test: {p[0, 1]=}')
    # # F = t_dif ** 2
    # # print(f'{F[0, 1]=}')
    #
    # # t, p = stats.ttest_rel(sn_inc_conn[:, 0, 0, 1], sn_inc_conn[:, 1, 0, 1],
    # #                        nan_policy='omit')
    # # print(t**2)
    # # quit()
    # t = stats.norm.ppf(1 - p)

    # F = t_dif ** 2


    atlas = get_atlas(schaefer=False)
    fp2name = {'obj4_fMRI': 'obj. encoding',
               'con3_fMRI': 'conc. retrieval',
               'vis3_fMRI': 'vis. retrieval'}
    plot_connectivity(z,
                      atlas['ticks'],
                      atlas['tick_labels'],
                      atlas['tick_lows'],
                      no_avg=True,
                      title=f'scene betas | paired t-test, effect of '
                            f'{kwargs["key"]} (YA + OA)',
                      vmin=0, vmax=4,
                      cbar_label='F-value')


def do_NBS(threshold=0.9, min_cluster_size=50, alpha_thresh=.05):
    # TODO: Anova?
    fp = 'scn7_fMRI'
    # fp = 'bl3_fMRI'
    # kwargs = {'fp': fp,
    #           'split': False,
    #           'key': 'inc_hit_hit',
    #           'key_vals': ('10', '11')
    #           }

    kwargs = {'fp': fp,
              'split': False,
              'key': 'con_hit',
              'key_vals': (False, True),
              'combine_regions': True
              }
    # kwargs = {'fp': fp,
    #           'split': False,
    #           'key': 'living',
    #           'key_vals': (False, True)
    #           }
    # kwargs = {'fp': fp, 'split': False,
    #           'key': 'vis_hit',
    #           'key_vals': (False, True),
    #           'odd_even': False
    #           }
    sn_inc_conn, sn_conn, age2idxs, sn_inc_activity = \
        pickle_wrap(None, load_FC_for_Lifu, kwargs=kwargs, verbose=1,
                    easy_override=False, cache_dir='cache')
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

    p12_graph = p2_graph
    z12_graph = z2_graph

    # SD12 = ((SD1_graph ** 2) * (N1_graph - 1) +
    #         (SD2_graph ** 2) * (N2_graph - 1)) / \
    #        (N1_graph + N2_graph - 2)
    # SE12 = np.sqrt(SD12 * (1 / N1_graph + 1 / N2_graph))
    # t12_graph = (M1_graph - M2_graph) / SE12
    # p12_graph = 1 - stats.t.cdf(t12_graph, N1_graph + N2_graph - 2)
    # z12_graph = -stats.norm.ppf(p12_graph)
    # # p12_graph = p12_graph_
    p12_graph = np.min([p12_graph, 1 - p12_graph], axis=0)
    bonf_correction = .05 / 30135 * 1000
    z_bonf = stats.norm.ppf(bonf_correction)
    print(f'{z_bonf=:.3f}')
    # z12_graph[p12_graph > bonf_correction] = np.nan
    p1_graph = np.min([p1_graph, 1 - p1_graph], axis=0)
    p2_graph = np.min([p2_graph, 1 - p2_graph], axis=0)


    # z12_graph[p12_graph > .05] = np.nan

    # p1_graph = np.min([p1_graph, 1 - p1_graph], axis=0)
    # // 2 accounts for bottom triangle
    num_signif = np.sum(p12_graph < alpha_thresh) // 2
    signif_graph = p12_graph < alpha_thresh
    signif_z_graph = z12_graph.copy()
    signif_z_graph[~signif_graph] = np.nan
    # fig, axs = plt.subplots(1, 2)
    # plt.sca(axs[0])

    atlas = get_atlas(schaefer=False)
    fp2name = {'obj4_fMRI': 'obj. encoding',
               'con3_fMRI': 'conc. retrieval',
               'vis3_fMRI': 'vis. retrieval',
               'obj7_fMRI': 'obj. encoding',
               'scn7_fMRI': 'scene encoding'}
    plot_connectivity(z1_graph,
                      atlas['ticks'],
                      atlas['tick_labels'],
                      atlas['tick_lows'],
                      no_avg=True,
                      title=f'{fp2name[kwargs["fp"]]} | paired t-test, effect of '
                            f'{kwargs["key"]} (YA + OA)',
                      vmin=-4, vmax=4,
                      cbar_label='t-value')
    # alpha_thresh = .05
    # num_signif = np.sum(p < alpha_thresh) // 2
    num_correction = .05 / alpha_thresh
    num_ROIs = sn_inc_conn.shape[2]
    num_edges = num_ROIs * (num_ROIs - 1) / 2
    expected_FP = num_edges * alpha_thresh
    alpha_thresh_z = stats.norm.ppf(alpha_thresh)

    # plt.hist(F.flatten(), bins=100, range=(0, 10))
    # plt.show()
    # quit()
    print(f'{alpha_thresh=} (z = {alpha_thresh_z:.2f}), {num_correction=} | '
          f'{num_edges=}, {expected_FP=:.2f}')
    print(f'\tNumber of significant edges: {num_signif=}')
    quit()
    # quit()
    # plt.imshow(z12_graph)
    # plt.title('Main effect of memory')
    # plt.title('z-graph')
    # plt.colorbar()
    # # plt.sca(axs[1])
    # # plt.imshow(signif_z_graph, interpolation='none', cmap='RdBu')
    # # plt.title('p-graph')
    # plt.show()
    # quit()

    num_correction = .05 / alpha_thresh
    num_ROIs = sn_inc_conn.shape[2]
    num_edges = num_ROIs * (num_ROIs - 1) / 2
    expected_FP = num_edges * alpha_thresh * 2
    alpha_thresh_z = stats.norm.ppf(alpha_thresh)
    print(f'{alpha_thresh=} (z = {alpha_thresh_z:.2f}), {num_correction=} | '
          f'{num_edges=}, {expected_FP=:.2f}')
    print(f'\tNumber of significant edges: {num_signif=}')
    quit()

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



def ttest_modularity():
    fp = 'obj7_fMRI'
    kwargs = {'fp': fp,
              'split': False,
              'key': 'inc',
              'key_vals': (1, 3)
              }

    sn_inc_conn, sn_conn, age2idxs, sn_inc_activity = \
        pickle_wrap(None, load_FC_for_Lifu, kwargs=kwargs, verbose=1,
                    easy_override=False, cache_dir='cache')

    M1_graph, SD1_graph, SE1_graph, N1_graph, t1_graph, p1_graph, z1_graph = \
        get_stats_graphs(sn_inc_conn[age2idxs[1], 0, :, :],
                         sn_inc_conn[age2idxs[1], 1, :, :])
    M2_graph, SD2_graph, SE2_graph, N2_graph, t2_graph, p2_graph, z2_graph = \
        get_stats_graphs(sn_inc_conn[age2idxs[2], 0, :, :],
                         sn_inc_conn[age2idxs[2], 1, :, :])
    threshold = .95
    partitions, matrix_mask = get_main_partitions(t2_graph, coords=None,
                                                  plot=True,
                        threshold=threshold, fn_str='', overlapping=False,
                        dir_out=f'OA2_ttest_modules_-thr{threshold}.png')
    quit()
    fp = 'obj7_fMRI'
    kwargs = {'fp': fp,
              'split': False,
              'key': 'inc_hit_hit',
              'key_vals': (False, True)
              }

    sn_inc_conn_mem, _, _, _ = \
        pickle_wrap(None, load_FC_for_Lifu, kwargs=kwargs, verbose=1,
                    easy_override=False, cache_dir='cache')
    sn_inc_conn_mem[:, :, matrix_mask == 0] = np.nan
    for i, p0 in enumerate(partitions):
        pmat = get_partition_matrix(sn_inc_conn_mem, p0)
        M_conn = np.nanmean(pmat, axis=(-2, -1))
        for age in [1, 2]:
            M_conn_age = M_conn[age2idxs[age], :]
            t, p = stats.ttest_rel(M_conn_age[:, 1], M_conn_age[:, 0],
                                   nan_policy='omit')
            print(f'partition {i}, {age=}: {t=:.2f}')

    # quit()
    #
    #
    # threshold = .95
    # partitions, matrix_mask = get_main_partitions(-t1_graph, coords=None,
    #                                               plot=False,
    #                     threshold=threshold, fn_str='', overlapping=False,
    #                     dir_out=f'YA_ttest_modules_-thr{threshold}.png')


if __name__ == '__main__':
    # ttest_modularity()
    # plot_M()
    do_NBS()
    # ANOVA_edges()




