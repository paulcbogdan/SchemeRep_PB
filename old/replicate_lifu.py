import numpy as np

from old.modularity import get_main_partitions, get_partition_matrix, get_BNA_coords
from vendor.old.network_funcs import load_FC_for_Lifu, get_ylim_settings, calculate_within_between, reconfiguration
from utils import pickle_wrap


def Fig15_Table2_analyses(fp='obj_fMRI', threshold=0.9):
    labels = ['Inc', 'Neu', 'Con']
    kwargs = {'fp': fp, 'split': False, 'key': 'inc', 'key_vals': (1, 2, 3)}
    sn_inc_conn, sn_conn, age2idxs, sn_inc_activity = pickle_wrap(None,  #f'cache/FC_data_{fp}.pkl',
                                                                  load_FC_for_Lifu, kwargs=kwargs,
                                                                  verbose=1, easy_override=False,
                                                                  cache_dir='../cache')
    partitions, top_edges_mat = get_main_partitions(sn_conn, plot=False,
                                                    threshold=threshold,
                                                    fn_str=fp)
    sn_inc_conn[:, :, ~top_edges_mat] = np.nan

    # if fp == 'obj_fMRI':

    # else:
    i2name = {0.9: {0: 'PFC+', 1: 'MTL+'},
              0.95: {0: 'MTL+', 2: 'dPFC', 5: 'vPFC'}} # 3: 'PCun', 4: 'SM',
    ps_good = [partitions[i] for i in i2name[threshold]]
    y_low_div, y_high_div = get_ylim_settings(sn_inc_conn, age2idxs, ps_good,
                                              do_division=True)
    y_low, y_high = get_ylim_settings(sn_inc_conn, age2idxs, ps_good,
                                      do_division=False)
    # print(f'{y_low=}, {y_high=}')
    # quit()
    for i, p in enumerate(partitions):
        if i not in i2name[threshold]:
            continue
        if len(p) < 5:
            continue
        print('-'*50)
        name = i2name[threshold][i]
        print(f'Partition: {name} ({i})')
        calculate_within_between(sn_inc_conn, age2idxs, p, labels,
                                 do_division=True, suptitle=name,
                                 y_low=y_low_div, y_high=y_high_div)
        calculate_within_between(sn_inc_conn, age2idxs, p, labels,
                                 do_division=False, suptitle=name,
                                 y_low=y_low, y_high=y_high)


def Fig16a_analyses(fp='obj_fMRI', threshold=0.9, memory_type='vis_hit'):
    if memory_type == 'vis_hit':
        labels = ['Vis hit', 'Vis miss']
    elif memory_type == 'con_hit':
        labels = ['Con hit', 'Con miss']
    elif memory_type == 'hit_hit':
        labels = ['Both hit', 'Either miss']
    else:
        raise ValueError(f'Unknown memory type: {memory_type=}')
    kwargs = {'fp': fp, 'split': False, 'key': memory_type,
              'key_vals': (False, True)}
    sn_inc_conn, sn_conn, age2idxs, sn_inc_activity = pickle_wrap(None,
                                                                  load_FC_for_Lifu, kwargs=kwargs,
                                                                  verbose=1, easy_override=False,
                                                                  cache_dir='../cache')
    partitions, top_edges_mat = get_main_partitions(sn_conn, plot=False,
                                                    threshold=threshold)
    sn_inc_conn[:, :, ~top_edges_mat] = np.nan
    i2name = {0.9: {0: 'PFC+', 1: 'MTL+'},
              0.95: {0: 'MTL+', 2: 'dPFC', 5: 'vPFC'}}
    ps_good = [partitions[i] for i in i2name[threshold]]
    y_low, y_high = get_ylim_settings(sn_inc_conn, age2idxs, ps_good,
                                      do_division=True)
    for i, p in enumerate(partitions):
        if i not in i2name[threshold]:
            continue
        if len(p) < 5:
            continue
        print('-'*50)
        name = i2name[threshold][i]
        print(f'Partition: {name} ({i})')
        reconfiguration(sn_inc_conn, age2idxs, p,
                        title=f'{name}, {memory_type}')
        # quit()
        calculate_within_between(sn_inc_conn, age2idxs, p, labels,
                                 y_low, y_high,
                                 do_division=True, suptitle=name)


def Fig17_analyses(fp='obj_fMRI',
                   threshold=0.9, plot=True):
    kwargs = {'fp': fp, 'split': False, 'key': 'vis_hit',
              'key_vals': (False, True)}
    sn_inc_conn, sn_conn, age2idxs, sn_inc_activity = pickle_wrap(None,
                                                                  load_FC_for_Lifu, kwargs=kwargs,
                                                                  verbose=1, easy_override=False,
                                                                  cache_dir='../cache')


    partitions, top_edges_mat = get_main_partitions(sn_conn, plot=False,
                                                    threshold=threshold)
    PFC_partition = partitions[0]

    sn_inc_conn = get_partition_matrix(sn_inc_conn, PFC_partition)
    sn_conn = get_partition_matrix(sn_conn, PFC_partition)
    PFC_coords = [coord for i, coord in enumerate(get_BNA_coords())
                  if i in PFC_partition]
    sub_partitions, top_edges_mat = get_main_partitions(sn_conn, plot=True,
                                                        threshold=threshold,
                                                        fn_str='PFC_')



if __name__ == '__main__':
    THRESHOLD = 0.9
    FP = 'obj_fMRI'
    # Fig15_Table2_analyses(fp=FP, threshold=THRESHOLD)
    # Fig16a_analyses(fp=FP, threshold=THRESHOLD, memory_type='vis_hit')
    Fig16a_analyses(fp=FP, threshold=THRESHOLD, memory_type='con_hit')
    # Fig16a_analyses(fp=FP, threshold=THRESHOLD, memory_type='hit_hit')

