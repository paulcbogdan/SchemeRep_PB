import warnings

import numpy as np
import pandas as pd
from connsearch.report import plot_ROI_scores
from tqdm import tqdm

from atlas_utils import get_atlas
from old.modularity import get_main_partitions, get_BNA_coords, get_binary_matrix
from old.classifiers import partition_classifier, partition_group_clf
from old.network_funcs import load_FC_for_Lifu
from utils import pickle_wrap

warnings.filterwarnings('ignore',
                        message='invalid value encountered in divide')


def generic_prep(kwargs, threshold=0.9):
    if isinstance(kwargs['key'], str):
        kwargs['key'] = (kwargs['key'],)
    elif len(kwargs['fp']) > 1:
        kwargs['fp_all'] = True
    if isinstance(kwargs['fp'], str):
        kwargs['fp'] = (kwargs['fp'],)
    elif len(kwargs['fp']) > 1:
        kwargs['fp_all'] = True

    sn_inc_activity = []
    sn_inc_conn = []
    sn_conn = []
    for key in kwargs['key']:
        for fp in kwargs['fp']:
            kwargs_ = kwargs.copy()
            kwargs_['fp'] = fp
            kwargs_['key'] = key
            sn_inc_conn_, sn_conn_, age2idxs, sn_inc_activity_ = \
                pickle_wrap(None, load_FC_for_Lifu, kwargs=kwargs_, verbose=1,
                            easy_override=False, cache_dir='../cache')
            sn_inc_conn.append(sn_inc_conn_)
            sn_conn.append(sn_conn_)
            sn_inc_activity.append(sn_inc_activity_)
            print(f'{sn_conn_.shape=}')
            print(f'{sn_inc_activity_.shape=}')
    sn_inc_conn = np.mean(sn_inc_conn, axis=0)
    sn_conn = np.mean(sn_conn, axis=0)
    sn_inc_activity = np.concatenate(sn_inc_activity, axis=3)
    print(f'{sn_inc_conn.shape=}')
    print(f'{sn_conn.shape=}')
    print(f'{sn_inc_activity.shape=}')

    sn_inc_activity = np.array(sn_inc_activity)
    atlas_name = kwargs['atlas_name'] if 'atlas_name' in kwargs else 'BNA'
    fp = f'cache/obj4_partitions_test_{atlas_name}_thr{threshold}.pkl'
    print(f'Partitions: {fp=}')
    coords = get_BNA_coords(atlas_name)
    partitions, top_edges_mat = pickle_wrap(fp, lambda: get_main_partitions(
        sn_conn, plot=True, threshold=threshold, coords=coords),
                                            easy_override=False)
    # print(top_edges_mat.shape)
    # quit()
    sn_inc_conn_unthresh = sn_inc_conn.copy()
    try:
        sn_inc_conn[:, :, ~top_edges_mat] = np.nan
    except IndexError: # laziness, need to make compatible with combine regions
        pass
    if atlas_name == 'BNA':
        i2name = {0.9: {0: 'PFC', 1: 'SM', 2: 'Vis', 3: 'MTL',
                        4: 'pSM', 5: 'PCun', 6: 'Limbic'},
                  0.95: {0: 'PFC', 1: 'Vis', 2: 'SM', 3: 'MTL',
                         4: 'pSM', 5: 'PCun'}}
    elif atlas_name == 'schaefer':
        i2name = {0.9: {0: 'PFC', 1: 'Vis', 2: 'SM', 3: 'Parietal',
                        4: 'pSM'},
                  0.95: {0: 'PFC', 1: 'Vis', 2: 'SM', 3: 'Parietal',
                         4: 'pSM', 5: 'MTL', 6: 'VAN'},
                  0.99: {0: 'VAN', 1: 'SM', 2: 'IPL', 3: 'dPFC',
                         4: 'Vis', 5: 'Precuneus', 6: 'MTG',
                         7: 'vPFC', 8: 'iVis'}}
    i2name = i2name[threshold]

    M_conn = np.nanmean(sn_inc_conn,
                        axis=tuple(range(len(sn_inc_conn.shape[:-2]))))
    matrix_binary, matrix_mask = get_binary_matrix(M_conn,
                                                     threshold=threshold)
    # top_edges_mat
    return partitions, sn_inc_activity, age2idxs, matrix_mask, i2name



def shuffle(sn_inc_activity):
    shuffler = np.array(list(range(sn_inc_activity.shape[1])))
    for i in range(sn_inc_activity.shape[0]):
        for k in range(sn_inc_activity.shape[3]):
            np.random.shuffle(shuffler)
            sn_inc_activity[i, :, :, k] = sn_inc_activity[i, shuffler, :, k]

def loop_over_ROIs(threshold=0.9, group=True):
    fp = 'obj4_fMRI'
    # fp = 'vis3_fMRI'
    # fp = ('obj4_fMRI', 'bl3_fMRI')


    kwargs = {'fp': fp, 'split': False,
              'key': 'inc',
              'key_vals': (1, 3),
              'odd_even': False,
              }
    kwargs = {'fp': fp, 'split': False,
              'key': 'inc',
              # 'atlas_name': 'schaefer',
              'atlas_name': 'BNA',
              'key_vals': (1, 3),
              'odd_even': False,
              }
    # fp = ('con3_fMRI', 'vis3_fMRI')
    # kwargs = {'fp': fp, 'split': False,
    #           # 'key': 'hit_hit',
    #           # 'key': ('vis_hit', 'con_hit'),
    #           'key': 'hit_hit',
    #           'atlas_name': 'BNA',
    #           'key_vals': (False, True),
    #           'odd_even': False,
    #           'pad_nan': True}
    # kwargs = {'fp': fp, 'split': False,
    #           'key': 'inc_match14_strict',
    #           'key_vals': (False, True),
    #           'odd_even': False,
    #           }
    # kwargs = {'fp': fp, 'split': False,
    #           'key': 'inc_match',
    #           'key_vals': (False, True),
    #           'odd_even': False,
    #           }


    # kwargs = {'fp': fp, 'split': False,
    #           # 'key': 'hit_hit',
    #           'key': 'vis_hit',
    #           'key_vals': (False, True),
    #           'odd_even': False,
    #           'pad_nan': True}

    # kwargs = {'fp': fp, 'split': False,
    #           # 'key': 'hit_hit',
    #           'key': ('vis_hit', 'con_hit'),
    #           'atlas_name': 'BNA',
    #           'key_vals': (False, True),
    #           'odd_even': False,
    # #           'pad_nan': True}
    # kwargs = {'fp': fp, 'split': False,
    #           # 'key': 'hit_hit',
    #           'key': 'vis_hit',# 'con_hit'),
    #           'atlas_name': 'schaefer',
    #           'key_vals': (False, True),
    #           'odd_even': False,
    #           'pad_nan': True}

    partitions, sn_inc_activity, age2idxs, top_edges_mat, i2name = \
        generic_prep(kwargs, threshold=threshold)
    # print(sn_inc_activity.shape)
    # quit()
    # shuffle(sn_inc_activity)

    trial_mapper = {}
    atlas = get_atlas(schaefer=kwargs['atlas_name'] == 'schaefer')
    ROIs = atlas['ROIs']
    coords = atlas['coords']
    ts_YA = []
    ts_OA = []
    for i, ROI in enumerate(tqdm(ROIs, desc='ROIwise classification')):
        edges = [(i, j) for j in range(len(ROIs)) if j != i]
        # edges = [(0, 1), (0, 2), (0, 3)]
        # shuffle(sn_inc_activity)
        # t_YA, t_OA = partition_classifier(sn_inc_activity, age2idxs, edges,
        #                                   trial_mapper=trial_mapper,
        #                                   stratification_strategy=1)
        fp_pkl = f'cache/p_clf_act_{threshold}_roi{ROI}_' \
                 f'{group}_ln{True}_' \
                 f'{kwargs["key"]}_{kwargs["fp"]}_{kwargs["key_vals"]}.pkl'
        if group:
            # Weirdly givingg so much with respectable accuracy...?
            f = lambda: partition_group_clf(sn_inc_activity, age2idxs, edges,
                                            trial_mapper=trial_mapper,
                                            stratification_strategy=1,
                                            linear=True)
            t_YA, t_OA = pickle_wrap(fp_pkl, f, easy_override=False)
        else:
            print('Conn:')
            f = lambda: partition_classifier(sn_inc_activity, age2idxs, edges,
                                             trial_mapper=trial_mapper,
                                             stratification_strategy=1,
                                             linear=True)
            t_YA, t_OA = pickle_wrap(fp_pkl, f, easy_override=False)
        print(f'{t_YA=:.2f}, {t_OA=:.2f}')






        print(f'{ROI} ({i}) | {t_YA=:.2f}, {t_OA=:.2f}')
        ts_YA.append(t_YA)
        ts_OA.append(t_OA)
        # if i > 5:
        #     break

    key = kwargs['key']
    coords = coords[:len(ts_OA)]
    fp = fr'result_pics/clf/YA_{key}.png'
    vmin = 0.51 if group else 2
    vmax = 0.6 if group else 4
    plot_ROI_scores(ts_YA, coords, fp_out=fp, show=True,
                    vmin=vmin, vmax=vmax, title='Younger adults')
    fp = fr'result_pics/clf/OA_{key}.png'
    plot_ROI_scores(ts_OA, coords, fp_out=fp, show=True,
                    vmin=vmin, vmax=vmax, title='Older adults')

def module_based_clf(threshold=0.95, group=True, linear=False):
    # fp = 'obj4_fMRI'
    fp = ('con3_fMRI', 'vis3_fMRI')
    fp = ('obj4_fMRI', 'bl3_fMRI')
    fp = 'obj4_fMRI'
    # fp = 'bl3_fMRI'
    # fp = 'vis3_fMRI'
    print(f'{group=} ({threshold=})')
    kwargs = {'fp': fp, 'split': False,
              'key': 'inc',
              # 'atlas_name': 'BNA',
              'key_vals': (1, 3),
              'odd_even': False,
              }

    # kwargs = {'fp': fp, 'split': False,
    #           'key': 'inc_match',
    #           'key_vals': (False, True),
    #           'odd_even': False,
    #           }
    # kwargs = {'fp': 'bl3_fMRI', 'split': False,
    #           'key': 'inc_match14',
    #           'key_vals': (False, True),
    #           'odd_even': False,
    #           }

    # kwargs = {'fp': fp, 'split': False,
    #           # 'key': 'hit_hit',
    #           'key': 'hit_hit',
    #           'atlas_name': 'schaefer',
    #           'key_vals': (False, True),
    #           'odd_even': False,
    #           'pad_nan': True}

    kwargs = {'fp': fp, 'split': False,
              # 'key': 'hit_hit',
              # 'key': ('vis_hit', 'con_hit'),
              'key': 'hit_hit',
              # 'atlas_name': 'BNA',
              'key_vals': (False, True),
              'odd_even': False,
              'pad_nan': True}

    # kwargs = {'fp': fp, 'split': False,
    #           # 'key': 'hit_hit',
    #           # 'key': ('inc_vis_hit', 'inc_con_hit'),
    #           'key': ('inc_hit_hit'),
    #           'atlas_name': 'BNA',
    #           'key_vals': ('30', '31'),
    #           'odd_even': False,
    #           'pad_nan': True}
    print(f'{linear=}')
    partitions, sn_inc_activity, age2idxs, top_edges_mat, i2name = \
        generic_prep(kwargs, threshold=threshold)

    age2idxs[3] = age2idxs[1] + age2idxs[2]
    trial_mapper = {}
    cols = []
    i2col = {}
    cnt = 0
    for i, p in enumerate(partitions):
        if i not in i2name:
            continue
        name = i2name[i]
        cols.append(name)
        i2col[i] = cnt
        cnt += 1
    cols = ['activity'] + cols
    # TODO: fix schaefer 997 ROI nonsense that doesn't align with coords

    results_YA = np.full((len(cols)-1, len(cols)), np.nan)
    results_OA = np.full((len(cols)-1, len(cols)), np.nan)

    i2name = {0: 'ventral_con', 1: 'dorsal_con'}
    partitions = [[35, 36, 37, 49, 69, 70, 71, 72, 73, 75, 76, 77, 79, 83, 87, 88, 90, 91, 93, 102, 103, 104, 105,
                   107, 108, 109, 110, 112, 115, 148, 149, 163, 166, 168, 171, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 202, 203, 204, 207, 211, 212, 213, 215, 217, 220, 221, 225, 226, 227, 229, 231, 236, 237, 243],
                  [0, 1, 2, 3, 4, 5, 10, 12, 13, 16, 18, 20, 21, 22, 23, 24, 26, 28, 29, 30, 31, 32, 33, 34, 38, 42, 43, 44, 45, 48, 50, 51, 62, 80, 81, 96, 99, 100, 101, 113, 119, 134, 136, 137, 140, 142, 146, 150, 152, 153, 162, 174, 175, 180, 181, 208, 209, 214, 233, 235, 238, 240, 242]]

    for i, p0 in enumerate(partitions):
        if i not in i2name:
            continue
        name0 = i2name[i]
        # if name0 == 'Vis':
        #     print(p0)
        #     quit()
        for j, p1 in enumerate(partitions):
            if j  not in i2name:
                continue
            name1 = i2name[j]
            # if j > i:
            #     continue
            if j < i:
                continue
            elif i == j:
                print(f'------- {i2name[i]} within -------')
                edges = [(a, b) for a in p0 for b in p0 if a < b]
                fp_pkl = f'cache/p_clf_act_{threshold}_{name0}_{name1}_' \
                         f'{group}_ln{linear}_' \
                         f'{kwargs["key"]}_{kwargs["fp"]}_{kwargs["key_vals"]}.pkl'
                print('Activity:')

                if not group:

                    f = lambda: partition_classifier(sn_inc_activity,
                                                     age2idxs,
                                                     edges,
                                                trial_mapper={},
                                                stratification_strategy=1,
                                                activity=True,
                                                     linear=linear)
                    t_YA_act, t_OA_act = pickle_wrap(fp_pkl, f,
                                                     easy_override=True)
                    results_YA[i2col[i], 0] = round(t_YA_act, 3)
                    results_OA[i2col[i], 0] = round(t_OA_act, 3)
                    print(f'{t_YA_act=:.2f}, {t_OA_act=:.2f}')
                else:
                    f = lambda: partition_group_clf(sn_inc_activity,
                                                    age2idxs,
                                                    edges,
                                              trial_mapper={},
                                              stratification_strategy=1,
                                                    acttivity=True,
                                                    linear=linear
                                                    )
                    t_YA_act, t_OA_act = pickle_wrap(fp_pkl, f,
                                                     easy_override=True)
                    results_YA[i2col[i], 0] = round(t_YA_act, 3)
                    results_OA[i2col[i], 0] = round(t_OA_act, 3)
                    print(f'{t_YA_act=:.2f}, {t_OA_act=:.2f}')
            else:
                print(f'------- {i2name[i]} x {i2name[j]} -------')
                edges = [(a, b) for a in p0 for b in p1 if a != b]
            fp_pkl = f'cache/p_clf_{threshold}_{name0}_{name1}_{group}_' \
                     f'ln{linear}_' \
                 f'{kwargs["key"]}_{kwargs["fp"]}_{kwargs["key_vals"]}.pkl'
            if group:
                # Weirdly givingg so much with respectable accuracy...?
                f = lambda: partition_group_clf(sn_inc_activity, age2idxs, edges,
                                              trial_mapper=trial_mapper,
                                              stratification_strategy=1,
                                                linear=linear)
                t_YA, t_OA = pickle_wrap(fp_pkl, f, easy_override=True)
            else:
                print('Conn:')
                f = lambda: partition_classifier(sn_inc_activity, age2idxs, edges,
                                                  trial_mapper=trial_mapper,
                                                  stratification_strategy=1,
                                                 linear=linear)
                t_YA, t_OA = pickle_wrap(fp_pkl, f, easy_override=True)
            print(f'{t_YA=:.2f}, {t_OA=:.2f}')

            print('-')
            results_YA[i2col[i], i2col[j]+1] = round(t_YA, 3)
            results_OA[i2col[i], i2col[j]+1] = round(t_OA, 3)
            # print(results_YA)
            # ts_YA.append(t_YA)
            # ts_OA.append(t_OA)
    title = f'{kwargs["key"]}, {fp}, {group=}'
    print(f'{kwargs=}')
    # print(f'{title=}')
    print(f'{linear=}')
    print('Young:')
    # increase print width
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    df_YA = pd.DataFrame(results_YA, index=cols[1:], columns=cols)
    print(df_YA)
    print('Old:')
    df_OA = pd.DataFrame(results_OA, index=cols[1:], columns=cols)
    print(df_OA)

if __name__ == '__main__':

    # run_clf()
    # loop_over_ROIs(0.95, group=False)
    module_based_clf(group=False)



