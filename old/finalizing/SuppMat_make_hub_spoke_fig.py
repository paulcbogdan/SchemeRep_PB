import os
import pathlib
path = pathlib.Path(__file__).parent.parent.resolve()
os.chdir(path)

import numpy as np
from matplotlib.colors import ListedColormap

from Study1A.load_Study1A_funcs import load_FC
from Study1A.plot_Fig2CD_partitions import get_VD_PA_partitions
from Utils.atlas_funcs import get_atlas
from nilearn import plotting

from Study1A.plot_Fig2AB_matrices import get_beta_graph
# from old.network_funcs import load_FC_for_Lifu
# from old_Apr6.ttest_mat import get_stats_graphs

import os
from Utils.pickle_wrap_funcs import pickle_wrap
import matplotlib.pyplot as plt

# from vendor_partitioning import get_vendor_partitions

# os.chdir(r'H:\PycharmProjects_H\SchemeRep')


def plot_hub_spoke(fp='obj7_fMRI', combine_regions=False, regr=False,
                   all_black=False, only_cortical=True):
    kwargs = {'fp': fp,
              'split': False,
              'key': 'inc',
              'key_vals': (1, 2, 3),
              }
    if combine_regions:
        kwargs['combine_regions'] = True
    sn_inc_conn, sn_conn, age2idxs, sn_inc_activity = \
        pickle_wrap(load_FC_for_Lifu, None, kwargs=kwargs,
                    easy_override=False, verbose=1,
                    cache_dir='cache')

    atlas = get_atlas(combine_regions=combine_regions,
                      combine_bilateral=False)


    if only_cortical:
        bad_rois = {'Amyg', 'Hipp', 'Str', 'Tha'}
        bad_j = [j for j, roi in enumerate(atlas['ROI_regions'])
                 if roi in bad_rois]
        sn_inc_conn[..., bad_j, :] = np.nan
        sn_inc_conn[..., :, bad_j] = np.nan

    if regr:
        t_graph = get_beta_graph(sn_inc_conn)
    else:
        _, _, _, _, t_graph, _, z_graph = \
            get_stats_graphs(sn_inc_conn[:, 0, :, :],
                             sn_inc_conn[:, 2, :, :])


    comp_graph = sn_inc_conn[:, 0, :, :] > sn_inc_conn[:, 2, :, :]

    # print(comp_graph.shape)
    # print(np.nan > np.nan)
    # t_graph = (np.nanmean(comp_graph, axis=0) - 0.5)
    if only_cortical:
        bad_rois = {'Amyg', 'Hipp', 'Str', 'Tha'}
        bad_j = [j for j, roi in enumerate(atlas['ROI_regions'])
                 if roi in bad_rois]
        t_graph[bad_j, :] = np.nan
        t_graph[:, bad_j] = np.nan

    num_nans = np.sum(np.isnan(t_graph[31]))
    # num_non_nans = np.sum(~np.isnan(t_graph[31]))
    # print(f'{num_nans=}')
    # quit()

    i = 77

    rois = [30, 76, 142, 202]
    # rois = [43, 51]
    # rois = [0, 3, 24]
    # rois = [0, 3, 24, 30, 31, 43, 51, 73, 76, 81, 99, 123, 142, 152, 173, 188,
    #         198, 202, 206, 207]

    # rois = [137]
    # rois = [73, 76]

    # rois = np.arange(40)

    # rois = [123, 142, 152, 173,]

    # rois = list(range(140, 147))

    # rois = np.arange(z_graph.shape[0])

    # rois = [24, 30, 31, 76]#, 142, 202]
    # rois = [roi - 1 for roi in rois]

    # for i in np.arange(t_graph.shape[0]):
    #     for j in np.arange(t_graph.shape[0]):
    #         if (i in rois ) and (j in rois ):
    #             continue
    #         t_graph[i, j] = np.nan
    #         continue

            # even_i = (i // 2) * 2
            # odd_i = even_i + 1
            # even_j = (j // 2) * 2
            # odd_j = even_j + 1
            # if ((even_i in rois or odd_i in rois) and
            #         (even_j in rois or odd_j in rois)):
            #     continue
            # t_graph[i, j] = np.nan

            # if i not in rois or j not in rois:
            #     t_graph[i, j] = np.nan

    # rois = [24, 30, 31]

    # rois = np.arange(20, 35)
    #
    #
    # t_graph_pre = t_graph.copy()
    # for roi in rois:
    #     t_graph[roi, :] = np.nanmean(t_graph_pre[rois], axis=0)

    # for :


    # rois = [31]

    # rois = [0]
    for i in rois:
    # for i in [78, 79]:
        all_black = False
        fig = plt.figure(figsize=(3.5, 3.5))
        # fig = plt.figure(figsize=(6, 6))

        t_graph_ = t_graph.copy()
        if all_black:
            t_graph_[:, :] = 1
            # z_graph_[:, ::3] = 1
            t_graph_[:i] = 0
            t_graph_[i+1:] = 0
            t_graph_[:, i] = t_graph_[i, :]
            vabs = 1
            edge_kw = {'linewidth': 1.5, 'alpha': 0.25}
        else:
            t_graph_[:i] = 0
            t_graph_[i+1:] = 0
            t_graph_[:, i] = t_graph_[i, :]
            vabs = 3
            edge_kw = {'linewidth': 2,}
        # t_graph_ = np.sqrt(np.abs(t_graph_)) * np.sign(t_graph_)

        cmap = plt.cm.get_cmap('turbo')
        cmap = ListedColormap(cmap(np.linspace(0.1, .95, 1000)))

        plotting.plot_connectome(
            t_graph_,
            atlas['coords'],
            edge_threshold=1,
            edge_vmin=-vabs, edge_vmax=vabs,
            colorbar=False,
            # edge_cmap='bone_r' if all_black else 'turbo', # 'cold_hot', #
            # edge_cmap='cold_hot',
            edge_cmap=cmap,
            node_size=1 if all_black else 1.5,
            node_color='k',
            edge_kwargs=edge_kw,
            display_mode='x',
            figure=fig,
            # title=atlas['ROIs'][i] + f' {atlas["coords"][i]}',
        )
        # t_graph_[t_graph_ > 0] = 1
        # t_graph_[t_graph_ < 0] = -1

        # plotting.plot_connectome(
        #     t_graph_,
        #     atlas['coords'],
        #     edge_threshold=1.65,
        #     edge_vmin=-vabs, edge_vmax=vabs,
        #     # colorbar=True,
        #     edge_cmap='bone_r' if all_black else 'turbo',  # 'cold_hot', #
        #     # edge_cmap='RdYlBu_r',
        #     # edge_vmax=1.5, edge_vmin=-1.5,
        #     # edge_cmap='RdBu_r',
        #     node_size=1 if all_black else 1.5,
        #     node_color='k',
        #     edge_kwargs=edge_kw,
        #     display_mode='xz',
        #     figure=fig,
        #     title=atlas['ROIs'][i] + f' {atlas["coords"][i]}',
        # )
        # abs_max = np.nanmax(np.abs(z_graph_))
        # print(f'{i}: {abs_max=:.3f}')
        # plt.colorbar()
        fp_out = fr'result_pics/other/hub_spoke/_{atlas["ROIs"][i]}.png'
        plt.savefig(fp_out, dpi=300)
        plt.show()


def plot_vendor_hub_spoke(fp='obj7_fMRI', combine_regions=False,
                          only_cortical=True, regr=True):
    p_dorsal, p_ventral, p_d_ant, p_d_pos, p_v_ant, p_v_pos, matrix_mask = \
        get_VD_PA_partitions(age='healthy', anat=True, thr=.9, anat_ver=3)

    kwargs = {'fp': fp,
              'split': False,
              'key': 'inc',
              'key_vals': (1, 2, 3),
              }
    if combine_regions:
        kwargs['combine_regions'] = True
    # sn_inc_conn, sn_conn, age2idxs, sn_inc_activity = \
    #     pickle_wrap(load_FC_for_Lifu, None, kwargs=kwargs,
    #                 easy_override=False, verbose=1,
    #                 cache_dir='cache')

    kwargs = {'fp': 'obj7_fMRI',
              'key': 'inc',
              'atlas_name': 'BNA',
              'key_vals': (1, 2, 3),
              'get_df_sn': True
              }
    sn_inc_conn, sn_conn, age2idxs, sn_inc_activity, _ = \
        pickle_wrap(load_FC, None, kwargs=kwargs,
                    easy_override=False, verbose=1, cache_dir='cache')

    atlas = get_atlas(combine_regions=combine_regions,
                      combine_bilateral=False)


    if only_cortical:
        bad_rois = {'Amyg', 'Hipp', 'Str', 'Tha'}
        bad_j = [j for j, roi in enumerate(atlas['ROI_regions'])
                 if roi in bad_rois]
        sn_inc_conn[..., bad_j, :] = np.nan
        sn_inc_conn[..., :, bad_j] = np.nan

    # if regr:
    #     t_graph = get_beta_graph(sn_inc_conn)
    # else:
    #     _, _, _, _, t_graph, _, z_graph = \
    #         get_stats_graphs(sn_inc_conn[:, 0, :, :],
    #                          sn_inc_conn[:, 2, :, :])

    t_graph = get_beta_graph(sn_inc_conn)


    idxs = p_v_ant
    t_graph_ = t_graph.copy()
    for i in range(t_graph.shape[0]):
        if i in idxs:
            continue
        for j in range(t_graph.shape[0]):
            if j in idxs:
                continue
            t_graph_[i, j] = np.nan
            t_graph_[j, i] = np.nan

    plotting.plot_connectome(
        t_graph_,
        atlas['coords'],
        edge_threshold=2,
        # colorbar=True,
        edge_cmap='turbo',  # 'cold_hot', #
        # edge_cmap='RdYlBu_r',
        node_size= 1.5,
        node_color='k',
        edge_kwargs={'linewidth': 1.5,
                     'alpha': .5},
        display_mode='xz',
        # figure=fig,
    )
    # abs_max = np.nanmax(np.abs(z_graph_))
    # print(f'{i}: {abs_max=:.3f}')
    # plt.colorbar()
    fp_out = fr'result_pics/other/hub_spoke_{atlas["ROIs"][i]}.png'
    plt.savefig(fp_out, dpi=300)
    plt.show()



if __name__ == '__main__':
    plot_vendor_hub_spoke()
    # plot_hub_spoke()






