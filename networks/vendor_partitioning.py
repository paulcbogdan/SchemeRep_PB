
import os
os.chdir('E:\PycharmProjects_E\SchemeRep')

from pathlib import Path
from collections import defaultdict, Counter
from copy import copy

import numpy as np

from atlas_utils import get_atlas
from ttest_mat import get_stats_graphs
from old.modularity import get_main_partitions
from old.network_funcs import load_FC_for_Lifu
from utils import pickle_wrap
import matplotlib.pyplot as plt

def get_vendor_partitions_(sn_inc_conn, age2idxs, age: int | str=2, thr=.95,
                           flip=True, weighted=True, plot=False):
    age2idxs['healthy'] = age2idxs[1] + age2idxs[2]
    if weighted:
        M_YA, _, _, _, _, p_YA, z_YA = \
            get_stats_graphs(sn_inc_conn[age2idxs[1], 0, :, :],
                             sn_inc_conn[age2idxs[1], 1, :, :])
        M_OA, _, _, _, _, p_OA, z_OA = \
            get_stats_graphs(sn_inc_conn[age2idxs[2], 0, :, :],
                             sn_inc_conn[age2idxs[2], 1, :, :])
        z_both = (z_YA + z_OA) / 2
        z_both = (M_YA + M_OA) / 2
    else:
        M_both, _, _, _, _, p_both, z_both = \
            get_stats_graphs(sn_inc_conn[age2idxs[age], 0, :, :],
                             sn_inc_conn[age2idxs[age], 1, :, :])
    z_both = -z_both if flip else z_both

    atlas = get_atlas()
    labels = atlas['labels']
    bad_labels = {'Str', 'Tha'}
    for i, label in enumerate(labels):
        for bad_label in bad_labels:
            if bad_label in label:
                z_both[i, :] = np.nan
                z_both[:, i] = np.nan
                break
    # plt.imshow(z_both)
    # plt.show()
    # quit()

    age2str = {1: 'YA', 2: 'OA', 'healthy': 'healthy'}
    weighted_str = '_W' if weighted else ''
    dir_out = f'result_pics/vendor/' \
              f'ttest_mod_{age2str[age]}_thr{thr}_flip{flip}{weighted_str}'
    if flip:
        title_extra = f' (Congruent > incongruent)'
    else:
        title_extra = f' (Incongruent > Congruent)'
    partitions, matrix_mask = \
        get_main_partitions(z_both, coords=None, plot=plot, threshold=thr,
                            fn_str='', overlapping=False,
                            dir_out_full=dir_out, title_extra=title_extra)

    return partitions, matrix_mask



def get_vendor_partitions(sn_inc_conn=None, age2idxs=None,
                          age: int | str='healthy',
                          thr=.95, flip=True, anat=False,
                          weighted=False, scrub=False,
                          plot=False, easy_override=False):
    if anat:
        return get_anat_vendor_partitions(plot=plot)
    if sn_inc_conn is None or age2idxs is None:
        kwargs = {'fp': 'obj7_fMRI',
                  'key': 'inc',
                  'atlas_name': 'BNA',
                  'key_vals': (1, 3),
                  }
        sn_inc_conn, sn_conn, age2idxs, sn_inc_activity = \
            pickle_wrap(None, load_FC_for_Lifu, kwargs=kwargs, verbose=1,
                        easy_override=False, cache_dir='cache')

    fp = f'cache/{age}_ttest_modules_thr{thr}_flip{flip}_{weighted}_n65.pkl'
    partitions, matrix_mask = \
        pickle_wrap(fp, lambda: get_vendor_partitions_(sn_inc_conn=sn_inc_conn,
                                                       age2idxs=age2idxs,
                                                       age=age, thr=thr,
                                                       flip=flip,
                                                       weighted=weighted,
                                                       plot=plot),
                    easy_override=easy_override)
    atlas = get_atlas()
    coords = atlas['coords']
    p_dorsal, p_ventral = partitions[0], partitions[1]
    p_d_ant, p_d_pos = anterior_posterior_split(p_dorsal, coords)
    p_v_ant, p_v_pos = anterior_posterior_split(p_ventral, coords)
    assert len(p_d_ant) + len(p_d_pos) == len(p_dorsal)
    assert len(p_v_ant) + len(p_v_pos) == len(p_ventral)
    if scrub:
        anat_str = '_anat' if anat else ''
        age_str = f'_YA' if age == 1 else '_OA' if age == 2 else \
            '_all_age' if age == 'healthy' else ''
        bonus_str = f'{anat_str}{age_str}'
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




def get_anat_vendor_partitions(plot=False):
    def labels2idxs(target):
        return [i for i, label in enumerate(labels) if
                any([l in label for l in target])]

    atlas = get_atlas()
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
    # quit()
    for i, (coord, label) in enumerate(zip(coords, labels)):
        for key in split_keys:
            if key in label:
                if coord[1] > split2median[key]:
                    labels[i] = f'{key}_a'
                else:
                    labels[i] = f'{key}_p'

    p_d_ant_labels = ['IFG', 'SFG', 'MFG', 'OrG']
    p_d_pos_labels = ['IPL', 'Pcun', 'PCC']
    p_v_ant_labels = ['ATL', 'PhG_a', 'ITG_a', 'MTG_a', 'STG_a', 'FuG_a']
    p_v_pos_labels = ['EVC', 'LOC', 'ITG_p', 'MTG_p', 'FuG_p']
    p_d_ant = labels2idxs(p_d_ant_labels)
    p_d_pos = labels2idxs(p_d_pos_labels)
    p_v_ant = labels2idxs(p_v_ant_labels)
    p_v_pos = labels2idxs(p_v_pos_labels)

    p_dorsal = p_d_ant + p_d_pos
    p_ventral = p_v_ant + p_v_pos
    matrix_mask = np.ones((246, 246), dtype=bool)

    if plot:
        bonus_str = '_anat'
        scrub_p(p_d_ant, p_d_pos, p_v_ant, p_v_pos, plot=True,
                bonus_str=bonus_str)

    return p_dorsal, p_ventral, p_d_ant, p_d_pos, p_v_ant, p_v_pos, matrix_mask






def scrub_p(p_d_ant, p_d_pos, p_v_ant, p_v_pos, plot=False,
            bonus_str=''):
    atlas = get_atlas()
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
            label = atlas['labels'][i].split(' ')[1].split('_')[0]
            quadrant2labels[quadrant].append(label)
            # print(f'{i}, {quadrant}: {label}')
            node_sizes.append(5)
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
        if label not in valid_labels[quadrant]:
            idx_to_quadrant[i] = f'mislabeled_{quadrant}'
            node_sizes[i] = 1

    if plot:
        dir_out = r'E:\PycharmProjects_E\SchemeRep\result_pics\vendor'
        Path(dir_out).mkdir(exist_ok=True, parents=True)
        fn_glass = fr'quads_original{bonus_str}.png'
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
            fp_glass = fr'{dir_out}\quads_scrubbing{bonus_str}.png'
            plot_glassbrain(idx_to_quadrant, edges, edge_weights, fp_glass,
                            coords, node_size=node_sizes_copy, linewidths=15,
                            network_colors=network_colors,)

            # from PIL import Image
            # Image.open(fp_glass).show()

            fp_glass = fr'{dir_out}\quads_scrubbed{bonus_str}.png'
            plot_glassbrain(idx_to_quadrant, edges, edge_weights, fp_glass,
                            coords, node_size=node_sizes, linewidths=15,
                            network_colors=network_colors,)

            # from PIL import Image
            # Image.open(fp_glass).show()

    f = lambda i: 'mislabeled' not in idx_to_quadrant[i]
    p_d_ant_new = list(filter(f, p_d_ant))
    p_d_pos_new = list(filter(f, p_d_pos))
    p_v_ant_new = list(filter(f, p_v_ant))
    p_v_pos_new = list(filter(f, p_v_pos))
    return p_d_ant_new, p_d_pos_new, p_v_ant_new, p_v_pos_new
    # else:
    #     return p_d_ant, p_d_pos, p_v_ant, p_v_pos

if __name__ == '__main__':
    THRESHOLD = 1.65
    get_vendor_partitions(age='healthy', flip=True, plot=True, scrub=True,
                          easy_override=True, thr=THRESHOLD)
    get_vendor_partitions(age=2, flip=True, anat=False, plot=True,
                          scrub=True, easy_override=True, thr=THRESHOLD)
    get_vendor_partitions(age=1, flip=True, anat=False, plot=True,
                          scrub=True, easy_override=True, thr=THRESHOLD)
    get_vendor_partitions(age='healthy', flip=False, anat=False, plot=True,
                          scrub=True, easy_override=True, thr=THRESHOLD)
    # get_vendor_partitions(age='healthy', flip=True, anat=True, plot=True,
    #                       scrub=True)
    # save_vendor_csv(p_d_ant, p_d_pos, p_v_ant, p_v_pos, scrub=False, anat=True)

    # p_dorsal, p_ventral, p_d_ant, p_d_pos, p_v_ant, p_v_pos, matrix_mask = \
    #     get_vendor_partitions(age='healthy', flip=True, scrub=True, plot=True)
    # save_vendor_csv(p_d_ant, p_d_pos, p_v_ant, p_v_pos, scrub=True, anat=False)



