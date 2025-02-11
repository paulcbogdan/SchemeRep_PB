import os
import pathlib


path = pathlib.Path(__file__).parent.parent.resolve()
os.chdir(path)

import os
import matplotlib.pyplot as plt

from scipy import stats

from Study1A.modularity_funcs import get_main_partitions
from Study1A.load_Study1A_funcs import load_FC

from collections import defaultdict
import numpy as np

from Utils.atlas_funcs import get_atlas
from Utils.pickle_wrap_funcs import pickle_wrap

np.random.seed(0) # to make the Louvain/Leiden (non-deterministic) output always the same


def get_regression_matrix(sn_inc_conn, flip=True, nans=True):
    sn_inc_conn = (sn_inc_conn -
                   np.nanmean(sn_inc_conn, axis=1)[:, None, :, :])
    n_sn = sn_inc_conn.shape[0]
    n_roi = sn_inc_conn.shape[-1]

    sn_conn = sn_inc_conn.reshape(-1, n_roi, n_roi)
    trils = np.tril_indices(n_roi, k=-1)
    sn_flat = sn_conn[:, trils[0], trils[1]]
    sn_flat = stats.zscore(sn_flat, axis=0, nan_policy='omit')

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
    z_both = np.full((n_roi, n_roi), np.nan)
    z_both[trils] = z
    z_both[trils[1], trils[0]] = z
    z_both = z_both if flip else -z_both
    var_beta_ = np.full((n_roi, n_roi), np.nan)
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


def get_VD_PA_partitions_(sn_inc_conn, age2idxs, age: int | str = 'healthy',
                          thr=.95, do_PA=True, plot=False,
                          combine_regions=False, regress=True):
    if regress:
        z_both = get_regression_matrix(sn_inc_conn, flip=do_PA)
    else:
        M_both, _, _, _, _, p_both, z_both = \
            get_1sample_ttest_matrix(sn_inc_conn[age2idxs[age], 0, :, :],
                                     sn_inc_conn[age2idxs[age], -1, :, :])
        z_both = -z_both if do_PA else z_both

    # plt.imshow(z_both)
    # plt.show()
    # quit()
    # schaefer = sn_inc_conn.shape[-1] % 100 == 0
    if sn_inc_conn.shape[-1] % 100 == 0:
        if sn_inc_conn.shape[-1] == 1000:
            schaefer = True
        else:
            schaefer = (True, sn_inc_conn.shape[-1])
    atlas = get_atlas(combine_regions=combine_regions, schaefer=schaefer)
    labels = atlas['labels']
    if not schaefer:
        bad_labels = {'Str', 'Tha', 'Amyg', 'Hipp', }
        for i, label in enumerate(labels):
            for bad_label in bad_labels:
                if bad_label in label:
                    z_both[i, :] = np.nan
                    z_both[:, i] = np.nan
                    break

    PA_VD_str = 'PA' if do_PA else 'VD'
    if isinstance(schaefer, tuple):
        schaef_str = f'schaef{schaefer[1]}_thr{thr}_'
    else:
        schaef_str = f'schaefer_thr{thr}_' if schaefer else ''
    cur_dir = os.getcwd()
    dir_out = f'{cur_dir}/result_pics/Fig2/{schaef_str}{PA_VD_str}_modules'

    partitions, matrix_mask = \
        get_main_partitions(z_both, coords=atlas['coords'], plot=plot,
                            threshold=thr, dir_out_full=dir_out, )


    for i, p in enumerate(partitions):
        labels = [atlas['labels'][i] for i in p]
        if len(labels) > 1:
            print(f'Partition {i}: {labels}')

    return partitions, matrix_mask


def get_VD_PA_partitions(sn_inc_conn=None, age2idxs=None,
                         age: int | str = 'healthy',
                         thr=.95, do_PA=True, anat=False,
                         plot=False, combine_regions=False,
                         anat_ver=3, regress=False,
                         schaefer=False):
    if schaefer and anat:
        atlas = get_atlas(combine_regions=combine_regions,
                          schaefer=schaefer)
        l = []
        keys = [('PFCl', 'PFClv', 'PFCv'),
                ('IPL', 'IPS', 'ParOper',
                 # 'ParOcc'
                 ),
                ('TempPole', 'AntTemp',
                 ),
                ('Striate', 'StriCal',
                 'ExStrInf', #'ExStrSup',
                 'ExStr' # LOC
                 )]
        for regions in keys:
            regions_idxs = []
            for region in regions:
                idxs = [i for i, ROI_region in enumerate(atlas['ROI_regions'])
                        if ROI_region == region]
                regions_idxs.extend(idxs)
            l.append(regions_idxs)
        p_d_ant, p_d_pos, p_v_ant, p_v_pos = l
        if not combine_regions:
            p_v_ant = ([258, 257, 241, 247, 246, 242] +
                       [252, 250, 249, 385, 399, 102, 265])
        p_dorsal = p_d_ant + p_d_pos
        p_ventral = p_v_ant + p_v_pos
        matrix_mask = np.ones((len(atlas['ROI_regions']),
                               len(atlas['ROI_regions'])),
                              dtype=bool)
        if plot:
            plot_quads_Fig2D(p_d_ant, p_d_pos, p_v_ant, p_v_pos,
                             combine_regions=combine_regions,
                             schaefer=schaefer)
        return p_dorsal, p_ventral, p_d_ant, p_d_pos, p_v_ant, p_v_pos, matrix_mask

    elif anat:
        return get_anat_VD_PA(plot=plot, combine_regions=combine_regions,
                              anat_ver=anat_ver)
    else:
        assert not combine_regions

    if sn_inc_conn is None or age2idxs is None:
        if isinstance(schaefer, tuple):
            atlas_name = schaefer
        elif schaefer:
            atlas_name = 'schaefer'
        else:
            atlas_name = 'BNA'

        kwargs = {'fp': 'obj7_fMRI',
                  'key': 'inc',
                  'atlas_name': atlas_name,
                  'key_vals': (1, 2, 3),
                  'get_df_sn': True
                  }
        sn_inc_conn, sn_conn, age2idxs, sn_inc_activity, _ = \
            pickle_wrap(load_FC, None, kwargs=kwargs,
                        easy_override=False, verbose=1, cache_dir='cache')
        age2idxs['healthy'] = age2idxs[1] + age2idxs[2]

    PA_VD_str = 'PA' if do_PA else 'VD'
    if isinstance(schaefer, tuple):
        schaef_str = f'_schaef{schaefer[1]}'
    else:
        schaef_str = '_schaefer' if schaefer else ''
    fp = (f'cache/{PA_VD_str}_modules_thresh{thr}{schaef_str}.pkl')

    partitions, matrix_mask = \
        pickle_wrap(lambda:
                    get_VD_PA_partitions_(sn_inc_conn=sn_inc_conn,
                                          age2idxs=age2idxs, age=age, thr=thr,
                                          do_PA=do_PA, plot=plot,
                                          combine_regions=combine_regions,
                                          regress=regress), fp,
                    easy_override=True)

    atlas = get_atlas(combine_regions=combine_regions,
                      schaefer=schaefer)
    coords = atlas['coords']
    p_dorsal, p_ventral = partitions[0], partitions[1]
    p_d_ant, p_d_pos = anterior_posterior_split(p_dorsal, coords)
    p_v_ant, p_v_pos = anterior_posterior_split(p_ventral, coords)
    assert len(p_d_ant) + len(p_d_pos) == len(p_dorsal)
    assert len(p_v_ant) + len(p_v_pos) == len(p_ventral)

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
        p_d_pos_labels = ['IPL', ]  # 'SPL' (SPL not supported by the con_reg)
        p_v_ant_labels = ['ATL', ]
        p_v_pos_labels = ['LOC', 'sOcG', 'EVC']
    elif anat_ver == 2:
        # Very minimal ROI lists. One region per list
        p_d_ant_labels = ['IFG']
        p_d_pos_labels = ['IPL']
        p_v_ant_labels = ['ATL']
        p_v_pos_labels = ['LOC', 'sOcG', 'OcG']  # This is just one ROI but different names
    elif anat_ver == 1:
        # Very expansive ROI lists, includes regions not implicated by classifier
        p_d_ant_labels = ['MFG', 'IFG', 'OrG', 'SFG', 'ACC']  #
        p_d_pos_labels = ['IPL', 'Pcun', 'SPL']  # 'SPL' (SPL not supported by the con_reg)
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
        plot_quads_Fig2D(p_d_ant, p_d_pos, p_v_ant, p_v_pos,
                         combine_regions=combine_regions)
    if make_csv:
        save_vendor_csv(p_d_ant, p_d_pos, p_v_ant, p_v_pos, scrub=False, anat=True)
    return p_dorsal, p_ventral, p_d_ant, p_d_pos, p_v_ant, p_v_pos, matrix_mask


def plot_quads_Fig2D(p_d_ant, p_d_pos, p_v_ant, p_v_pos,
                     combine_regions=False, schaefer=False):
    atlas = get_atlas(combine_regions=combine_regions,
                      schaefer=schaefer)


    idx_to_quadrant = {i: 'PD' for i in p_d_pos}
    idx_to_quadrant.update({i: 'PV' for i in p_v_pos})
    idx_to_quadrant.update({i: 'AD' for i in p_d_ant})
    idx_to_quadrant.update({i: 'AV' for i in p_v_ant})
    node_sizes = []
    # quadrant2labels = defaultdict(list)
    for i in range(len(atlas['labels'])):
        if i not in idx_to_quadrant:
            idx_to_quadrant[i] = 'N/A'
            node_sizes.append(0)
        else:
            node_sizes.append(10)

    from nichord import plot_glassbrain

    cur_dir = os.getcwd()
    if isinstance(schaefer, tuple):
        schaef_str = f'_schaef{schaefer[1]}'
    else:
        schaef_str = '_schaefer' if schaefer else ''
    fp_quad = fr'{cur_dir}\result_pics\Fig2\Fig2D_anat_quads{schaef_str}.png'

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
    print(f'{idx_to_quadrant}')
    print(len(idx_to_quadrant))
    # print(f'{idx_to_quadrant=}')
    # print(f'{node_sizes=}')
    # quit()

    plot_glassbrain(idx_to_quadrant, edges, edge_weights, fp_quad,
                    coords, node_size=node_sizes, linewidths=15,
                    network_colors=network_colors, )


if __name__ == '__main__':
    # Data-driven modules (Fig 2C)
    SCHAEFER = True
    THR = 0.995 if SCHAEFER else 0.95
    SCHAEFER = (True, 400)
    THR = 0.98
    # get_VD_PA_partitions(do_PA=True, plot=True, thr=THR, regress=True,
    #                      schaefer=SCHAEFER)
    # get_VD_PA_partitions(do_PA=False, plot=True, thr=THR, regress=True,
    #                      schaefer=SCHAEFER)
    # quit()

    # Four anatomical quadrants (Fig 2D)
    get_VD_PA_partitions(plot=True, anat_ver=3, anat=True,
                         schaefer=SCHAEFER)
