
import os

os.chdir(r'H:\PycharmProjects_H\SchemeRep')

from pathlib import Path
from collections import defaultdict, Counter
from copy import copy

import numpy as np

from atlas_utils import get_atlas
from old_Apr6.ttest_mat import get_stats_graphs
from old.modularity import get_main_partitions
from old.network_funcs import load_FC_for_Lifu
from utils import pickle_wrap, stdize
import matplotlib.pyplot as plt

def do_regression(sn_inc_conn, flip=True, nans=True):
    sn_inc_conn = (sn_inc_conn -
                   np.nanmean(sn_inc_conn, axis=1)[:, None, :, :])
    n_sn = sn_inc_conn.shape[0]
    n_roi = sn_inc_conn.shape[-1]
    # print(sn_inc_conn.shape)
    # quit()

    sn_conn = sn_inc_conn.reshape(-1, n_roi, n_roi)
    trils = np.tril_indices(n_roi, k=-1)
    sn_flat = sn_conn[:, trils[0], trils[1]]
    sn_flat = stdize(sn_flat, axis=0, nans=nans)
    regressors = np.array([[-1, 0, 1] * n_sn]).T

    XTX_inv = np.linalg.inv(np.dot(regressors.T, regressors))
    XTX_invX = np.dot(XTX_inv, regressors.T)
    # print(XTX_inv)
    # XTX_invX = np.nansum(XTX_inv * regressors.T)
    # XTX_invX = XTX_inv * regressors.T
    # print(test)
    # print(XTX_invX)
    # print(test.shape)
    # print(XTX_invX.shape)
    # quit()

    if nans:
        betas = np.nansum(XTX_invX * sn_flat.T, axis=1)[None, :]
    else:
        betas = np.dot(XTX_invX, sn_flat)

    Y_pred = np.dot(regressors, betas)
    residual = sn_flat - Y_pred
    # print(sn_flat.T.shape)
    # print(betas.shape)
    # quit()
    # print(sn_flat.T[4278, :])
    # print(Y_pred[:, 4278])
    # quit()

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
    # print(np.sum(var_beta == 0))

    z_both = np.full((246, 246), np.nan)
    z_both[trils] = z
    z_both[trils[1], trils[0]] = z
    z_both = z_both if flip else -z_both
    # num_nans = np.sum(np.isnan(betas))
    var_beta_ =  np.full((n_roi, n_roi), np.nan)
    # print(sigma_s)
    var_beta_[trils] = np.nansum(residual ** 2, axis=0)
    # print(var_beta_[93, :])

    # print(residual[:, 4278])
    # quit()

    # test = np.nansum(residual ** 2, axis=0)
    # for i in range(10_000):
    #     if test[i] == 0:
    #         print(i)
    #         quit()

    # plt.imshow(z_both)
    # plt.colorbar()
    # plt.show()
    # plt.imshow(betas)
    # plt.show()

    # plt.imshow(var_beta)
    # plt.show()

    # print(num_nans)
    # quit()

    return z_both

def get_vendor_partitions_(sn_inc_conn, age2idxs, age: int | str=2, thr=.95,
                           flip=True, weighted=True, plot=False,
                           combine_regions=False, regress=True):

    if regress:
        z_both = do_regression(sn_inc_conn, flip=flip)

    elif weighted:
        M_YA, _, _, _, _, p_YA, z_YA = \
            get_stats_graphs(sn_inc_conn[age2idxs[1], 0, :, :],
                             sn_inc_conn[age2idxs[1], -1, :, :])
        M_OA, _, _, _, _, p_OA, z_OA = \
            get_stats_graphs(sn_inc_conn[age2idxs[2], 0, :, :],
                             sn_inc_conn[age2idxs[2], -1, :, :])
        z_both = (z_YA + z_OA) / 2
        # z_both = (M_YA + M_OA) / 2
        z_both = -z_both if flip else z_both
    else:
        M_both, _, _, _, _, p_both, z_both = \
            get_stats_graphs(sn_inc_conn[age2idxs[age], 0, :, :],
                             sn_inc_conn[age2idxs[age], -1, :, :])
        z_both = -z_both if flip else z_both

    # age2idxs['healthy'] = age2idxs[1] + age2idxs[2]

    atlas = get_atlas(combine_regions=combine_regions)
    labels = atlas['labels']
    bad_labels = {'Str', 'Tha', 'Amyg', 'Hipp',} #
    # bad_labels = set()
    for i, label in enumerate(labels):
        for bad_label in bad_labels:
            if bad_label in label:
                z_both[i, :] = np.nan
                z_both[:, i] = np.nan
                break
    # plt.imshow(z_both)
    # plt.show()
    # quit()

    num_non_nans = np.sum(~np.isnan(sn_inc_conn[:, 0, 0, 1]))

    age2str = {1: 'YA', 2: 'OA', 'healthy': 'healthy'}
    weighted_str = '_W' if weighted else ''
    comb_str = '_comb' if combine_regions else ''
    regr_str = '_regr' if regress else ''
    root = r'H:\PycharmProjects_H\SchemeRep\result_pics'
    dir_out = f'{root}/ttest_modules/' \
              f'flip{flip}_thr{thr}_{age2str[age]}{comb_str}{weighted_str}' \
              f'{regr_str}_n{num_non_nans}'
    regression_str = ' [regression]' if regress else ''
    if flip:
        title_extra = f' (Congruent > incongruent){regression_str}'
    else:
        title_extra = f' (Incongruent > Congruent){regression_str}'
    partitions, matrix_mask = \
        get_main_partitions(z_both, coords=atlas['coords'], plot=plot,
                            threshold=thr,
                            fn_str='', overlapping=False,
                            dir_out_full=dir_out,
                            title_extra=title_extra,
                            )


    for i, p in enumerate(partitions):
        labels = [atlas['labels'][i] for i in p]
        print(f'Partition {i}: {labels}')

    return partitions, matrix_mask



def get_vendor_partitions(sn_inc_conn=None, age2idxs=None,
                          age: int | str='healthy',
                          thr=.95, flip=True, anat=False,
                          weighted=False, scrub=False,
                          plot=False, easy_override=False,
                          combine_regions=False,
                          anat_ver=1,
                          regress=False):
    if anat:
        return get_anat_vendor_partitions(plot=plot,
                                          combine_regions=combine_regions,
                                          anat_ver=anat_ver)
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
        # print(sn_inc_conn.shape)
        # quit()


    comb_str = f'_comb' if combine_regions else ''
    regr_str = f'_regr' if regress else ''
    assert not (weighted and regress)
    fp = (f'cache/{age}_ttest_modules_thr{thr}_flip{flip}{comb_str}_'
          f'{weighted}{regr_str}_n65.pkl')

    partitions, matrix_mask = \
        pickle_wrap(lambda:
                    get_vendor_partitions_(sn_inc_conn=sn_inc_conn,
                                           age2idxs=age2idxs, age=age, thr=thr,
                                           flip=flip, weighted=weighted,
                                           plot=plot,
                                           combine_regions=combine_regions,
                                           regress=regress), fp,
                    easy_override=True)
    # print(partitions)
    # quit()
    atlas = get_atlas(combine_regions=combine_regions)
    coords = atlas['coords']
    # print(len(coords))
    p_dorsal, p_ventral = partitions[0], partitions[1]
    p_d_ant, p_d_pos = anterior_posterior_split(p_dorsal, coords)
    p_v_ant, p_v_pos = anterior_posterior_split(p_ventral, coords)
    assert len(p_d_ant) + len(p_d_pos) == len(p_dorsal)
    assert len(p_v_ant) + len(p_v_pos) == len(p_ventral)
    if scrub:
        anat_str = '_anat' if anat else ''
        age_str = f'_YA' if age == 1 else '_OA' if age == 2 else \
            '_all_age' if age == 'healthy' else ''
        thresh_str = f'_thr{thr}'
        bonus_str = f'{anat_str}{age_str}{thresh_str}{comb_str}'
        p_d_ant, p_d_pos, p_v_ant, p_v_pos = scrub_p(p_d_ant, p_d_pos,
                                                     p_v_ant, p_v_pos,
                                                     plot=plot,
                                                     bonus_str=bonus_str)

    return p_dorsal, p_ventral, p_d_ant, p_d_pos, p_v_ant, p_v_pos, matrix_mask

def save_vendor_csv(p_d_ant, p_d_pos, p_v_ant, p_v_pos, scrub, anat):
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
    fp_out_csv = fr'docs/vendor_partitions{scrubbed_str}{anat_str}.csv'
    df.to_csv(fp_out_csv, index=False)


def anterior_posterior_split(p_dorsal, coords):
    p_dorsal_ys = [coords[i][1] for i in p_dorsal]
    p_dorsal_y_med = np.median(p_dorsal_ys)
    p_dorsal_ant = [i for i in p_dorsal if coords[i][1] > p_dorsal_y_med]
    p_dorsal_pos = [i for i in p_dorsal if coords[i][1] <= p_dorsal_y_med]
    return p_dorsal_ant, p_dorsal_pos




def get_anat_vendor_partitions(plot=False, anat_ver=1, combine_regions=False,
                               ):
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

    if anat_ver == 6:
        p_d_ant_labels = ['MFG', 'IFG', 'OrG', 'SFG', 'ACC'] #
        p_d_pos_labels = ['IPL', 'Pcun', 'SPL'] # 'SPL' (SPL not supported by the con_reg)
        p_v_ant_labels = ['ATL', 'STG', 'MTG', 'ITG', 'FuG']
        p_v_pos_labels = ['LOC', 'sOcG', 'EVC']
    elif anat_ver == 5:
        p_d_ant_labels = ['MFG', 'IFG', ] # 'OrG', 'SFG'
        p_d_pos_labels = ['IPL', 'SPL', 'Pcun'] # 'Pcun', 'SPL' (SPL not supported by the con_reg)
        p_v_ant_labels = ['ATL', 'MTG', 'STG'] # ATL is 2x STG, 4x MTG, 2x ITG
        p_v_pos_labels = ['LOC', 'sOcG', 'EVC']
    elif anat_ver == 4:
        p_d_ant_labels = ['MFG', 'IFG', ] # 'OrG', 'SFG'
        p_d_pos_labels = ['IPL', 'Pcun'] # 'SPL' (SPL not supported by the con_reg)
        p_v_ant_labels = ['ATL', 'STG']
        p_v_pos_labels = ['LOC', 'sOcG', 'EVC']
    elif anat_ver == 3: # Consistent with rbf classifiers (PoG, STG, pSTS discared)
        p_d_ant_labels = ['MFG', 'IFG']
        p_d_pos_labels = ['IPL', ] # 'SPL' (SPL not supported by the con_reg)
        p_v_ant_labels = ['ATL', ]
        p_v_pos_labels = ['LOC', 'sOcG', 'EVC']
    elif anat_ver == 2:
        p_d_ant_labels = ['IFG']
        p_d_pos_labels = ['IPL']
        p_v_ant_labels = ['ATL']
        p_v_pos_labels = ['LOC', 'sOcG', 'OcG']
    elif anat_ver == 1:
        p_d_ant_labels = ['IFG', 'SFG', 'MFG',] #  'OrG'
        p_d_pos_labels = ['IPL', 'SPL', 'Pcun', ] # 'PCC'
        p_v_ant_labels = ['ATL', 'PhG_a',
                          'ITG_a', 'MTG_a', 'STG_a', 'FuG_a']
        p_v_pos_labels = ['sOcG', 'EVC', 'LOC', 'PhG_p',
                          'ITG_p', 'MTG_p', 'STG_p','FuG_p']
    else:
        p_d_ant_labels = ['IFG', 'SFG', 'MFG', 'OrG']
        p_d_pos_labels = ['IPL', 'Pcun', 'PCC']
        p_v_ant_labels = ['ATL', 'PhG_a', 'ITG_a', 'MTG_a', 'STG_a', 'FuG_a']
        p_v_pos_labels = ['EVC', 'LOC', 'ITG_p', 'MTG_p', 'FuG_p']
    p_d_ant = labels2idxs(p_d_ant_labels)
    p_d_pos = labels2idxs(p_d_pos_labels)
    p_v_ant = labels2idxs(p_v_ant_labels)
    # print(labels)
    # print(f'{p_v_ant_labels=}')
    # print(f'{p_v_ant=}')
    # quit()
    p_v_pos = labels2idxs(p_v_pos_labels)

    print(f'{len(p_d_ant)=}')
    print(f'{len(p_d_pos)=}')
    print(f'{len(p_v_ant)=}')
    print(f'{len(p_v_pos)=}')
    total = len(p_d_ant) + len(p_d_pos) + len(p_v_ant) + len(p_v_pos)
    print(f'{total=}')

    p_dorsal = p_d_ant + p_d_pos
    p_ventral = p_v_ant + p_v_pos
    matrix_mask = np.ones((246, 246), dtype=bool)

    if plot:
        bonus_str = '_anat'
        scrub_p(p_d_ant, p_d_pos, p_v_ant, p_v_pos, plot=True,
                bonus_str=bonus_str, combine_regions=combine_regions)

    # save_vendor_csv(p_d_ant, p_d_pos, p_v_ant, p_v_pos, scrub=False, anat=True)
    return p_dorsal, p_ventral, p_d_ant, p_d_pos, p_v_ant, p_v_pos, matrix_mask






def scrub_p(p_d_ant, p_d_pos, p_v_ant, p_v_pos, plot=False,
            bonus_str='', combine_regions=False):

    atlas = get_atlas(combine_regions=combine_regions)

    from nichord import plot_glassbrain
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
            # print(f'{i}, {quadrant}: {label}')
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
        label = atlas['labels'][i].split(' ')[1].split('_')[0]
        if label not in valid_labels[quadrant] and 'anat' not in bonus_str:
            idx_to_quadrant[i] = f'mislabeled_{quadrant}'
            node_sizes[i] = 1

    if plot:
        dir_out = r'H:\PycharmProjects_H\SchemeRep\result_pics\ttest_modules'
        Path(dir_out).mkdir(exist_ok=True, parents=True)
        if 'anat' in bonus_str:
            fn_glass = 'quads_anat.png'
        else:
            fn_glass = fr'quads_scrubbing{bonus_str}.png'
        fp_glass = fr'{dir_out}\{fn_glass}'
        # fp_glass = fr'{dir_out}\glass_first_scrub_colored.png'
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

        plot_glassbrain(idx_to_quadrant, edges, edge_weights, fp_glass,
                        coords, node_size=node_sizes, linewidths=15,
                        network_colors=network_colors,)

        for key in ['AD', 'AV', 'PD', 'PV']:
            network_colors[f'mislabeled_{key}'] = network_colors[key]
        node_sizes_copy = copy(node_sizes)
        scrubbed = False
        for i, size in enumerate(node_sizes):
            if size < max(node_sizes) and size > 0:
                node_sizes[i] = 0
                node_sizes_copy[i] = max(node_sizes)
                scrubbed = True

        if scrubbed:
            fp_glass = fr'{dir_out}\quads_original{bonus_str}.png'
            plot_glassbrain(idx_to_quadrant, edges, edge_weights, fp_glass,
                            coords, node_size=node_sizes_copy, linewidths=15,
                            network_colors=network_colors,)

            fp_glass = fr'{dir_out}\quads_scrubbed{bonus_str}.png'
            plot_glassbrain(idx_to_quadrant, edges, edge_weights, fp_glass,
                            coords, node_size=node_sizes, linewidths=15,
                            network_colors=network_colors,)

    f = lambda i: 'mislabeled' not in idx_to_quadrant[i]
    p_d_ant_new = list(filter(f, p_d_ant))
    p_d_pos_new = list(filter(f, p_d_pos))
    p_v_ant_new = list(filter(f, p_v_ant))
    p_v_pos_new = list(filter(f, p_v_pos))
    return p_d_ant_new, p_d_pos_new, p_v_ant_new, p_v_pos_new

if __name__ == '__main__':
    # get_vendor_partitions(age='healthy', flip=True, plot=True,
    #                       scrub=True, easy_override=True, anat=True,
    #                       anat_ver=3)
    REGRESS = True
    for THRESHOLD in [.95]:
        get_vendor_partitions(age='healthy', flip=True, plot=True,
                              scrub=True, easy_override=True, thr=THRESHOLD,
                              combine_regions=False, regress=REGRESS)
        get_vendor_partitions(age='healthy', flip=False, plot=True,
                              scrub=True, easy_override=True, thr=THRESHOLD,
                              combine_regions=False, regress=REGRESS)
