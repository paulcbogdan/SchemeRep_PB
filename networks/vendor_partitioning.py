import os
from collections import defaultdict

import numpy as np

from atlas_utils import get_atlas
from network_based_statistic import get_stats_graphs
from old.modularity import get_main_partitions
from old.network_funcs import load_FC_for_Lifu
from utils import pickle_wrap


os.chdir('C:\PycharmProjects_C\SchemeRep')

def get_vendor_partitions_(sn_inc_conn, age2idxs, age: int | str=2, thr=.95,
                           flip=False):
    age2idxs['healthy'] = age2idxs[1] + age2idxs[2]
    M1_graph, SD1_graph, SE1_graph, N1_graph, t2_graph, p1_graph, z2_graph = \
        get_stats_graphs(sn_inc_conn[age2idxs[age], 0, :, :],
                         sn_inc_conn[age2idxs[age], 1, :, :])
    z2_graph = -z2_graph if flip else z2_graph

    age2str = {1: 'YA', 2: 'OA', 'healthy': 'healthy'}
    dir_out = f'nichord_plots/vendor/' \
              f'ttest_mod_{age2str[age]}_thr{thr}_flip{flip}'
    if flip:
        title_extra = f' (Congruent > incongruent)'
    else:
        title_extra = f' (Incogruent > Congruent)'
    partitions, matrix_mask = \
        get_main_partitions(z2_graph, coords=None, plot=True, threshold=thr,
                            fn_str='', overlapping=False,
                            dir_out_full=dir_out, title_extra=title_extra)
    # print('MADE AND PLOTTED')
    # quit()
    return partitions, matrix_mask



def get_vendor_partitions(sn_inc_conn=None, age2idxs=None,
                          age: int | str='healthy',
                          thr=.95, flip=True, anat=False):
    if anat:
        return get_anat_vendor_partitions()
    if sn_inc_conn is None or age2idxs is None:
        kwargs = {'fp': 'obj7_fMRI',
                  'key': 'inc',
                  'atlas_name': 'BNA',
                  'key_vals': (1, 3),
                  }
        sn_inc_conn, sn_conn, age2idxs, sn_inc_activity = \
            pickle_wrap(None, load_FC_for_Lifu, kwargs=kwargs, verbose=1,
                        easy_override=False, cache_dir='cache')

    fp = f'cache/{age}_ttest_modules_thr{thr}_flip{flip}.pkl'
    partitions, matrix_mask = \
        pickle_wrap(fp, lambda: get_vendor_partitions_(sn_inc_conn=sn_inc_conn,
                                                       age2idxs=age2idxs,
                                                       age=age, thr=thr,
                                                       flip=flip),
                    easy_override=True)
    atlas = get_atlas()
    coords = atlas['coords']
    p_dorsal, p_ventral = partitions[0], partitions[1]
    p_d_ant, p_d_pos = anterior_posterior_split(p_dorsal, coords)
    p_v_ant, p_v_pos = anterior_posterior_split(p_ventral, coords)
    assert len(p_d_ant) + len(p_d_pos) == len(p_dorsal)
    assert len(p_v_ant) + len(p_v_pos) == len(p_ventral)
    print(f'{p_d_ant=}')
    print(f'{p_d_pos=}')
    print(f'{p_v_ant=}')
    print(f'{p_v_pos=}')
    quit()
    return p_dorsal, p_ventral, p_d_ant, p_d_pos, p_v_ant, p_v_pos, matrix_mask


def anterior_posterior_split(p_dorsal, coords):
    p_dorsal_ys = [coords[i][1] for i in p_dorsal]
    p_dorsal_y_med = np.median(p_dorsal_ys)
    p_dorsal_ant = [i for i in p_dorsal if coords[i][1] > p_dorsal_y_med]
    p_dorsal_pos = [i for i in p_dorsal if coords[i][1] <= p_dorsal_y_med]
    return p_dorsal_ant, p_dorsal_pos




def get_anat_vendor_partitions():
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
    return p_dorsal, p_ventral, p_d_ant, p_d_pos, p_v_ant, p_v_pos, None



if __name__ == '__main__':
    get_vendor_partitions(age='healthy', flip=False)