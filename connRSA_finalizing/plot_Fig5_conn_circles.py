from Utils.atlas_funcs import get_atlas
from Utils.pickle_wrap_funcs import pickle_wrap
from connRSA_finalizing.plot_Fig5_conn import run_IC_analysis, plot_FC_mat
import numpy as np
from nilearn import plotting

def plot_conn_circles(ERS=True, FC=False):
    if FC:
        corrs, _ = (
            pickle_wrap(plot_FC_mat, None, easy_override=False))
    else:
        corrs = run_IC_analysis(ERS, get_M=True)

    corrs = np.nanmean(corrs, axis=0)
    atlas = get_atlas(combine_regions=False, lifu_labels=True)
    coords = np.array(atlas['coords'], dtype=float)

    M_corrs = np.zeros(len(atlas['coords']))
    idx_to_label = {}

    region2network = {'SFG': 'PFC', 'MFG': 'PFC', 'IFG': 'PFC', 'OrG': 'PFC',
                      'SPL': 'Parietal', 'IPL': 'Parietal', 'Pcun': 'Parietal',
                      'EVC': 'Occipital', 'LOC': 'Occipital', 'sOcG': 'Occipital',
                      'ITG': 'ITL', 'FuG': 'ITL', 'PhG': 'ITL', 'ATL': 'ITL'}

    for LR in [True, False,]:
        coords_ = coords.copy()
        if LR:
            coords_[::2] += float('inf')
        else:
            coords_[1::2] += float('inf')
        for i in range(len(atlas['coords'])):
            if LR:
                if i % 2 == 0:
                    continue
            else:
                if i % 2 == 1:
                    continue
            dists = np.linalg.norm(coords_[i] - coords_, axis=1)
            closest_ranking = np.argsort(dists)
            closest_ranking = closest_ranking[1:11]
            M_corr = np.mean(corrs[i, closest_ranking], axis=0)
            M_corrs[i] = M_corr
            region = atlas['ROI_regions'][i]
            if region in region2network:
                idx_to_label[i] = region2network[region]
            else:
                idx_to_label[i] = 'N/A'

    from nichord import plot_glassbrain

    network_colors = {'Occipital': 'dodgerblue', 'ITL': 'orange',
                      'Parietal': 'red', 'PFC': 'green',
                      'N/A': 'black'}
    edges = [(i, i) for i in range(len(coords))]
    edge_weights = [0] * len(edges)

    if ERS:
        fp_fig = f'result_pics/Figure5B_circle_ERS.png'
    elif FC:
        fp_fig = f'result_pics/Figure5C_conn_FC.png'
    else:
        fp_fig = f'result_pics/Figure5A_circle_NSM.png'

    if ERS:
        node_sizes = M_corrs * 9
    elif FC:
        node_sizes = M_corrs * 5
    else:
        node_sizes = M_corrs * 5
    node_sizes *= 1.1
    node_sizes **= 4
    node_sizes[np.isnan(node_sizes)] = np.nanmin(node_sizes)
    node_sizes[node_sizes < 1] = 1


    plot_glassbrain(idx_to_label, edges, edge_weights, fp_fig,
                    coords, node_size=node_sizes, linewidths=15,
                    network_colors=network_colors, )


if __name__ == '__main__':
    plot_conn_circles()
    plot_conn_circles(ERS=False)
    plot_conn_circles(ERS=False, FC=True)





