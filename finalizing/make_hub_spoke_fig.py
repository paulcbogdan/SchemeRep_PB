from atlas_utils import get_atlas
from nilearn import plotting

from finalizing.make_Fig3_matrix import get_beta_graph
from old.network_funcs import load_FC_for_Lifu
from old_Apr6.ttest_mat import get_stats_graphs

import os
from utils import pickle_wrap
import matplotlib.pyplot as plt

os.chdir(r'H:\PycharmProjects_H\SchemeRep')


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

    if regr:
        z_graph = get_beta_graph(sn_inc_conn)
    else:
        _, _, _, _, _, _, z_graph = \
            get_stats_graphs(sn_inc_conn[:, 0, :, :],
                             sn_inc_conn[:, 2, :, :])

    i = 77



    for i in [30, 76, 142, 202]:
    # for i in [78, 79]:
        all_black = False
        fig = plt.figure(figsize=(3.5, 3.5))

        z_graph_ = z_graph.copy()
        if all_black:
            z_graph_[:, :] = 1
            # z_graph_[:, ::3] = 1
            z_graph_[:i] = 0
            z_graph_[i+1:] = 0
            z_graph_[:, i] = z_graph_[i, :]
            vabs = 1
            edge_kw = {'linewidth': 1.5, 'alpha': 0.25}
        else:
            z_graph_[:i] = 0
            z_graph_[i+1:] = 0
            z_graph_[:, i] = z_graph_[i, :]
            vabs = 4
            edge_kw = {'linewidth': 1.5,}

        plotting.plot_connectome(
            z_graph_,
            atlas['coords'],
            edge_threshold=1.,
            edge_vmin=-vabs, edge_vmax=vabs,
            colorbar=False,
            edge_cmap='bone_r' if all_black else 'turbo', # 'cold_hot', #
            node_size=1 if all_black else 1.5,
            node_color='k',
            edge_kwargs=edge_kw,
            display_mode='x',
            figure=fig,
            # title=atlas['ROIs'][i] + f' {atlas["coords"][i]}',
        )
        # abs_max = np.nanmax(np.abs(z_graph_))
        # print(f'{i}: {abs_max=:.3f}')
        fp_out = fr'result_pics/other/hub_spoke/_{atlas["ROIs"][i]}.png'
        plt.savefig(fp_out, dpi=300)
        plt.show()




if __name__ == '__main__':
    plot_hub_spoke()






