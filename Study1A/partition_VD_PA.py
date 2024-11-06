import os
import pathlib
path = pathlib.Path(__file__).parent.parent.resolve()
os.chdir(path)


import os

from scipy import stats

from networks.old.modularity import get_main_partitions
from Study1A.load_data_Study1A import load_FC_for_Lifu

# os.chdir(r'/')

from pathlib import Path
from collections import defaultdict, Counter
from copy import copy

import numpy as np

from atlas_utils import get_atlas
from utils import pickle_wrap, stdize



from nilearn import datasets
# print(datasets.utils.get_data_dirs())
# quit()
dataset = datasets.fetch_atlas_yeo_2011()

# path = pathlib.Path(__file__).parent.parent.resolve()
# os.chdir(path)

# from nilearn import datasets
# datasets.get_data_dirs('cache')

def get_regression_matrix(sn_inc_conn, flip=True, nans=True):
    sn_inc_conn = (sn_inc_conn -
                   np.nanmean(sn_inc_conn, axis=1)[:, None, :, :])
    n_sn = sn_inc_conn.shape[0]
    n_roi = sn_inc_conn.shape[-1]

    sn_conn = sn_inc_conn.reshape(-1, n_roi, n_roi)
    trils = np.tril_indices(n_roi, k=-1)
    sn_flat = sn_conn[:, trils[0], trils[1]]
    sn_flat = stdize(sn_flat, axis=0, nans=nans)
    regressors = np.array([[-1, 0, 1] * n_sn]).T

    XTX_inv = np.linalg.inv(np.dot(regressors.T, regressors))
    XTX_invX = np.dot(XTX_inv, regressors.T)

    if nans:
        betas = np.nansum(XTX_invX * sn_flat.T, axis=1)[None, :]
    else:
        betas = np.dot(XTX_invX, sn_flat)

    Y_pred = np.dot(regressors, betas)
    residual = sn_flat - Y_pred

    n_sn = np.sum(np.any(~np.isnan(sn_inc_conn), axis=1),
                  axis=0)

    n_sn = n_sn[trils]

    if nans:
        sigma_s = np.nansum(residual ** 2, axis=0) / (n_sn * 2 - 2)
        ss_x = np.nansum(regressors ** 2, axis=0)
    else:
        sigma_s = np.sum(residual ** 2, axis=0) / (n_sn * 2 - 2)
        ss_x = np.sum(regressors ** 2, axis=0)
    var_beta = sigma_s / ss_x

    z = betas / np.sqrt(var_beta)
    z_both = np.full((246, 246), np.nan)
    z_both[trils] = z
    z_both[trils[1], trils[0]] = z
    z_both = z_both if flip else -z_both
    var_beta_ =  np.full((n_roi, n_roi), np.nan)
    var_beta_[trils] = np.nansum(residual ** 2, axis=0)

    return z_both

def get_1sample_ttest_matrix(graph0, graph1):
    dif_graph = graph0 - graph1

    M_graph = np.nanmean(dif_graph, axis=0)
    SD_graph = np.nanstd(dif_graph, axis=0)
    N_graph = np.nansum(~np.isnan(dif_graph), axis=0)
    SE_graph = SD_graph / (N_graph ** 0.5)
    t_graph = M_graph / SE_graph
    p_graph = stats.t.cdf(t_graph, N_graph - 1)
    z_graph = stats.norm.ppf(p_graph)
    p_graph = np.min([p_graph, 1 - p_graph], axis=0)
    return M_graph, SD_graph, SE_graph, N_graph, t_graph, p_graph, z_graph

def get_VD_PA_partitions_(sn_inc_conn, age2idxs, age: int | str='healthy',
                          thr=.95, do_PA=True, plot=False,
                          combine_regions=False, regress=True):

    if regress:
        z_both = get_regression_matrix(sn_inc_conn, flip=do_PA)
    else:
        M_both, _, _, _, _, p_both, z_both = \
            get_1sample_ttest_matrix(sn_inc_conn[age2idxs[age], 0, :, :],
                             sn_inc_conn[age2idxs[age], -1, :, :])
        z_both = -z_both if do_PA else z_both

    atlas = get_atlas(combine_regions=combine_regions)
    labels = atlas['labels']
    bad_labels = {'Str', 'Tha', 'Amyg', 'Hipp',}
    for i, label in enumerate(labels):
        for bad_label in bad_labels:
            if bad_label in label:
                z_both[i, :] = np.nan
                z_both[:, i] = np.nan
                break

    PA_VD_str = 'PA' if do_PA else 'VD'
    cur_dir = os.getcwd()
    dir_out = f'{cur_dir}/result_pics/Fig2/{PA_VD_str}_modules'
    regression_str = ' [regression]' if regress else ''
    if do_PA:
        title_extra = f' (Congruent > incongruent){regression_str}'
    else:
        title_extra = f' (Incongruent > Congruent){regression_str}'
    partitions, matrix_mask = \
        get_main_partitions(z_both, coords=atlas['coords'], plot=plot,
                            threshold=thr,
                            fn_str='', overlapping=False,
                            dir_out_full=dir_out,
                            title_extra=title_extra,)

    for i, p in enumerate(partitions):
        labels = [atlas['labels'][i] for i in p]
        print(f'Partition {i}: {labels}')

    return partitions, matrix_mask

def get_VD_PA_partitions(sn_inc_conn=None, age2idxs=None,
                         age: int | str='healthy',
                         thr=.95, do_PA=True, anat=False,
                         weighted=False, scrub=False,
                         plot=False, combine_regions=False,
                         anat_ver=3, regress=False):
    if anat:
        return get_anat_VD_PA(plot=plot, combine_regions=combine_regions,
                              anat_ver=anat_ver)
    else:
        assert not combine_regions

    if sn_inc_conn is None or age2idxs is None:
        kwargs = {'fp': 'obj7_fMRI',
                  'key': 'inc',
                  'split': False,
                  'atlas_name': 'BNA',
                  'key_vals': (1, 2, 3) if regress else (1, 3),
                  'get_df_sn': True,
                  'combine_regions': combine_regions
                  }
        sn_inc_conn, sn_conn, age2idxs, sn_inc_activity, _ = \
            pickle_wrap(load_FC_for_Lifu, None, kwargs=kwargs,
                        easy_override=False, verbose=1, cache_dir='cache')

    PA_VD_str = 'PA' if do_PA else 'VD'
    fp = (f'cache/{PA_VD_str}_modules_thresh{thr}.pkl')

    partitions, matrix_mask = \
        pickle_wrap(lambda:
                    get_VD_PA_partitions_(sn_inc_conn=sn_inc_conn,
                                          age2idxs=age2idxs, age=age, thr=thr,
                                          do_PA=do_PA, plot=plot,
                                          combine_regions=combine_regions,
                                          regress=regress), fp,
                    easy_override=True)

    # Code below would be useful for getting the ROIs of the data-driven modules
    #   The small number of ROIs outside the quadrants can be dropped with scrub=True
    #   This would reproduce the PE x Direction bars, but were not used for the report
    atlas = get_atlas(combine_regions=combine_regions)
    coords = atlas['coords']
    p_dorsal, p_ventral = partitions[0], partitions[1]
    p_d_ant, p_d_pos = anterior_posterior_split(p_dorsal, coords)
    p_v_ant, p_v_pos = anterior_posterior_split(p_ventral, coords)
    assert len(p_d_ant) + len(p_d_pos) == len(p_dorsal)
    assert len(p_v_ant) + len(p_v_pos) == len(p_ventral)
    if scrub:
        p_d_ant, p_d_pos, p_v_ant, p_v_pos = scrub_p(p_d_ant, p_d_pos,
                                                     p_v_ant, p_v_pos,)

    return p_dorsal, p_ventral, p_d_ant, p_d_pos, p_v_ant, p_v_pos, matrix_mask

def save_vendor_csv(p_d_ant, p_d_pos, p_v_ant, p_v_pos, scrub, anat):
    # Can output a .csv with the ROIs in each quadrant
    #   Not used for report
    atlas = get_atlas()
    l_out = []
    for i, (label, coord) in enumerate(zip(atlas['labels'], atlas['coords'])):
        d = {'label': label}
        if i in p_d_ant:
            quad = 'DorAnt'
        elif i in p_d_pos:
            quad = 'DorPos'
        elif i in p_v_ant:
            quad = 'VenAnt'
        elif i in p_v_pos:
            quad = 'VenPos'
        else:
            quad = None
        d['quadrant'] = quad
        d['x'] = coord[0]
        d['y'] = coord[1]
        d['z'] = coord[2]
        l_out.append(d)
    import pandas as pd
    df = pd.DataFrame(l_out)
    scrubbed_str = '_scrubbed' if scrub else ''
    anat_str = '_anat' if anat else ''
    fp_out_csv = fr'result_pics/vendor_partitions{scrubbed_str}{anat_str}.csv'
    df.to_csv(fp_out_csv, index=False)

def anterior_posterior_split(p_dorsal, coords):
    p_dorsal_ys = [coords[i][1] for i in p_dorsal]
    p_dorsal_y_med = np.median(p_dorsal_ys)
    p_dorsal_ant = [i for i in p_dorsal if coords[i][1] > p_dorsal_y_med]
    p_dorsal_pos = [i for i in p_dorsal if coords[i][1] <= p_dorsal_y_med]
    return p_dorsal_ant, p_dorsal_pos

def get_anat_VD_PA(plot=False, anat_ver=1, combine_regions=False,
                   make_csv=False):
    def labels2idxs(target):
        return [i for i, label in enumerate(labels) if
                any([l in label for l in target])]

    atlas = get_atlas(combine_regions=combine_regions)
    coords, labels = atlas['coords'], atlas['labels']
    split_keys = ['PhG', 'ITG', 'MTG', 'STG', 'FuG']
    split2l = defaultdict(list)
    for coord, label in zip(coords, labels):
        for key in split_keys:
            if key in label:
                split2l[key].append(coord)
    split2median = {}
    for key, l in split2l.items():
        split2l[key] = np.array(l)
        if len(split2l[key]) == 6:
            split2median[key] = np.sort(split2l[key][:, 1])[3] + .0001
        else:
            split2median[key] = np.median(split2l[key][:, 1])
    for i, (coord, label) in enumerate(zip(coords, labels)):
        for key in split_keys:
            if key in label:
                if coord[1] > split2median[key]:
                    labels[i] = f'{key}_a'
                else:
                    labels[i] = f'{key}_p'

    if anat_ver == 3:
        # ROI lists based on classifiers (PoG, STG, pSTS discarded)
        #   These are the lists best motivated by the task data
        p_d_ant_labels = ['MFG', 'IFG']
        p_d_pos_labels = ['IPL', ] # 'SPL' (SPL not supported by the con_reg)
        p_v_ant_labels = ['ATL', ]
        p_v_pos_labels = ['LOC', 'sOcG', 'EVC']
    elif anat_ver == 2:
        # Very minimal ROI lists. One region per list
        p_d_ant_labels = ['IFG']
        p_d_pos_labels = ['IPL']
        p_v_ant_labels = ['ATL']
        p_v_pos_labels = ['LOC', 'sOcG', 'OcG'] # This is just one ROI but different names
    elif anat_ver == 1:
        # Very expansive ROI lists, includes regions not implicated by classifier
        p_d_ant_labels = ['MFG', 'IFG', 'OrG', 'SFG', 'ACC'] #
        p_d_pos_labels = ['IPL', 'Pcun', 'SPL'] # 'SPL' (SPL not supported by the con_reg)
        p_v_ant_labels = ['ATL', 'STG', 'MTG', 'ITG', 'FuG']
        p_v_pos_labels = ['LOC', 'sOcG', 'EVC']
    else:
        raise ValueError

    p_d_ant = labels2idxs(p_d_ant_labels)
    p_d_pos = labels2idxs(p_d_pos_labels)
    p_v_ant = labels2idxs(p_v_ant_labels)
    p_v_pos = labels2idxs(p_v_pos_labels)

    p_dorsal = p_d_ant + p_d_pos
    p_ventral = p_v_ant + p_v_pos
    matrix_mask = np.ones((246, 246), dtype=bool)

    if plot:
        bonus_str = '_anat'
        plot_quads_Fig2D(p_d_ant, p_d_pos, p_v_ant, p_v_pos,
                         combine_regions=combine_regions, bonus_str=bonus_str)

    if make_csv:
        save_vendor_csv(p_d_ant, p_d_pos, p_v_ant, p_v_pos, scrub=False, anat=True)
    return p_dorsal, p_ventral, p_d_ant, p_d_pos, p_v_ant, p_v_pos, matrix_mask

def plot_quads_Fig2D(p_d_ant, p_d_pos, p_v_ant, p_v_pos,
                     combine_regions=False, bonus_str=''):

    atlas = get_atlas(combine_regions=combine_regions)

    idx_to_quadrant = {i: 'PD' for i in p_d_pos}
    idx_to_quadrant.update({i: 'PV' for i in p_v_pos})
    idx_to_quadrant.update({i: 'AD' for i in p_d_ant})
    idx_to_quadrant.update({i: 'AV' for i in p_v_ant})
    node_sizes = []
    quadrant2labels = defaultdict(list)
    for i in range(len(atlas['labels'])):
        if i not in idx_to_quadrant:
            idx_to_quadrant[i] = 'N/A'
            node_sizes.append(0)
        else:
            quadrant = idx_to_quadrant[i]
            label = atlas['labels'][i]
            print(f'Pre: {label=}')
            label = label.split(' ')[1].split('_')[0]
            quadrant2labels[quadrant].append(label)
            node_sizes.append(10)

    from nichord import plot_glassbrain

    cur_dir = os.getcwd()
    fp_quad = fr'{cur_dir}result_pics\Fig2\Fig2D_anat_quads.png'

    coords = atlas['coords']
    edges = [(i, i) for i in range(len(coords))]
    edge_weights = [0] * len(edges)
    network_colors = {'AD': 'red', 'AV': 'dodgerblue',
                      'PD': 'limegreen', 'PV': 'orange',
                      'mislabeled_AD': 'darkred',
                      'mislabeled_AV': 'darkblue',
                      'mislabeled_PD': 'darkgreen',
                      'mislabeled_PV': 'darkgoldenrod',
                      'N/A': 'black'}

    plot_glassbrain(idx_to_quadrant, edges, edge_weights, fp_quad,
                    coords, node_size=node_sizes, linewidths=15,
                    network_colors=network_colors, )
def scrub_p(p_d_ant, p_d_pos, p_v_ant, p_v_pos, combine_regions=False):
    atlas = get_atlas(combine_regions=combine_regions)

    idx_to_quadrant = {i: 'PD' for i in p_d_pos}
    idx_to_quadrant.update({i: 'PV' for i in p_v_pos})
    idx_to_quadrant.update({i: 'AD' for i in p_d_ant})
    idx_to_quadrant.update({i: 'AV' for i in p_v_ant})
    node_sizes = []
    quadrant2labels = defaultdict(list)
    for i in range(len(atlas['labels'])):
        if i not in idx_to_quadrant:
            idx_to_quadrant[i] = 'N/A'
            node_sizes.append(0)
        else:
            quadrant = idx_to_quadrant[i]
            label = atlas['labels'][i]
            print(f'Pre: {label=}')
            label = label.split(' ')[1].split('_')[0]
            quadrant2labels[quadrant].append(label)
            node_sizes.append(10)
    for k, v in quadrant2labels.items():
        print(f'{k}: {Counter(v)}')

    valid_labels = {'AD': {'IFG', 'SFG', 'MFG', 'OrG', 'ACC', 'PrG'},
                    'AV': {'ATL', 'PhG', 'Hipp', 'STG', 'ITG', 'MTG', 'FuG'}, # INS? AMY?
                    'PD': {'IPL', 'Pcun', 'PCC', 'SPL', 'sOcG'},
                    'PV': {'EVC', 'LOC', 'ITG', 'FuG', 'MTG'}}
    for i, quadrant in idx_to_quadrant.items():
        if quadrant == 'N/A':
            continue
        if label not in valid_labels[quadrant]:
            idx_to_quadrant[i] = f'mislabeled_{quadrant}'

    f = lambda i: 'mislabeled' not in idx_to_quadrant[i]
    p_d_ant_new = list(filter(f, p_d_ant))
    p_d_pos_new = list(filter(f, p_d_pos))
    p_v_ant_new = list(filter(f, p_v_ant))
    p_v_pos_new = list(filter(f, p_v_pos))
    return p_d_ant_new, p_d_pos_new, p_v_ant_new, p_v_pos_new



if __name__ == '__main__':
    get_VD_PA_partitions(age='healthy', do_PA=False, plot=True,
                         scrub=False, anat_ver=3, anat=True,
                         combine_regions=False, regress=True)

    for THRESHOLD in [.95]:
        get_VD_PA_partitions(age='healthy', do_PA=True, plot=True,
                             scrub=False, thr=THRESHOLD,
                             combine_regions=False, regress=True)
        get_VD_PA_partitions(age='healthy', do_PA=False, plot=True,
                             scrub=False, thr=THRESHOLD,
                             combine_regions=False, regress=True)
