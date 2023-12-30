import warnings

import numpy as np
import pandas as pd
from connsearch.report import plot_ROI_scores
from tqdm import tqdm

from atlas_utils import get_atlas
from modularity import get_partition_matrix, get_main_partitions, get_BNA_coords
from classifiers import partition_classifier, partition_group_clf
from network_funcs import load_FC_for_Lifu, reconfiguration, \
    analyze_subject_specific, subj_specific_repeated
from utils import pickle_wrap

warnings.filterwarnings('ignore',
                        message='invalid value encountered in divide')


def generic_prep(kwargs, threshold=0.9):
    if isinstance(kwargs['key'], str):
        kwargs['key'] = (kwargs['key'],)
    else:
        kwargs['fp_all'] = True
    if isinstance(kwargs['fp'], str):
        kwargs['fp'] = (kwargs['fp'],)
    else:
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
                            easy_override=False, cache_dir='cache')
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
    coords = get_BNA_coords(atlas_name)
    partitions, top_edges_mat = pickle_wrap(fp, lambda: get_main_partitions(
        sn_conn, plot=True, threshold=threshold, coords=coords),
                                            easy_override=False)
    sn_inc_conn_unthresh = sn_inc_conn.copy()
    sn_inc_conn[:, :, ~top_edges_mat] = np.nan
    if atlas_name == 'BNA':
        i2name = {0.9: {0: 'PFC', 1: 'SM', 2: 'Vis', 3: 'MTL',
                        4: 'pSM', 5: 'PCun', 6: 'Limbic'},
                  0.95: {0: 'PFC', 1: 'Vis', 2: 'SM', 3: 'MTL',
                         4: 'pSM', 5: 'PCun'}}
    elif atlas_name == 'schaefer':
        i2name = {0.9: {0: 'PFC', 1: 'Vis', 2: 'SM', 3: 'Parietal',
                        4: 'pSM'},
                  0.95: {0: 'PFC', 1: 'Vis', 2: 'SM', 3: 'Parietal',
                         4: 'pSM', 5: 'MTL', 6: 'VAN'}}
    i2name = i2name[threshold]

    return partitions, sn_inc_activity, age2idxs, top_edges_mat, i2name



def shuffle(sn_inc_activity):
    # print(sn_inc_activity[0, :, 0, :])
    # print(sn_inc_activity.shape)
    # quit()
    shuffler = np.array(list(range(sn_inc_activity.shape[1])))

    for i in range(sn_inc_activity.shape[0]):
        # for j in range(sn_inc_activity.shape[2]):
        # rand_splitter = np.random.choice(list(range(sn_inc_activity.shape[3])),
        #                                  size=sn_inc_activity.shape[3] // 2,
        #                                  replace=False)

        for k in range(sn_inc_activity.shape[3]):
            np.random.shuffle(shuffler)
            # if i in rand_splitter:
            #     shuffler = np.array([0, 1])
            # else:
            #     shuffler = np.array([1, 0])
            sn_inc_activity[i, :, :, k] = sn_inc_activity[i, shuffler, :, k]

def loop_over_ROIs(threshold=0.9):
    fp = 'obj4_fMRI'
    # fp = 'vis3_fMRI'
    # fp = ('obj4_fMRI', 'bl3_fMRI')


    kwargs = {'fp': fp, 'split': False,
              'key': 'inc',
              'key_vals': (1, 3),
              'odd_even': False,
              }
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

    # shuffle(sn_inc_activity)

    trial_mapper = {}
    atlas = get_atlas()
    ROIs = atlas['ROIs']
    coords = atlas['coords']
    ts_YA = []
    ts_OA = []
    for i, ROI in enumerate(tqdm(ROIs, desc='ROIwise classification')):
        edges = [(i, j) for j in range(len(ROIs)) if j != i]
        # edges = [(0, 1), (0, 2), (0, 3)]
        # shuffle(sn_inc_activity)
        t_YA, t_OA = partition_classifier(sn_inc_activity, age2idxs, edges,
                                          trial_mapper=trial_mapper,
                                          stratification_strategy=2)
        print(f'{ROI} ({i}) | {t_YA=:.2f}, {t_OA=:.2f}')
        ts_YA.append(t_YA)
        ts_OA.append(t_OA)
        # if i > 5:
        #     break

    key = kwargs['key']
    coords = coords[:len(ts_OA)]
    fp = fr'nichord_plots/ROI_clf/YA_{key}.png'
    plot_ROI_scores(ts_YA, coords, fp_out=fp, show=True,
                    vmin=2, vmax=4, title='Younger adults')
    fp = fr'nichord_plots/ROI_clf/OA_{key}.png'
    plot_ROI_scores(ts_OA, coords, fp_out=fp, show=True,
                    vmin=2, vmax=4, title='Older adults')

def module_based_clf(threshold=0.95, group=True):
    # fp = 'obj4_fMRI'
    fp = 'obj4_fMRI'#, 'bl3_fMRI')
    # fp = 'bl3_fMRI'
    # fp = 'vis3_fMRI'

    kwargs = {'fp': 'vis3_fMRI', 'split': False,
              'key': 'inc',
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
    #           'key': 'vis_hit',
    #           'atlas_name': 'BNA',
    #           'key_vals': (False, True),
    #           'odd_even': False,
    #           'pad_nan': True}

    # kwargs = {'fp': fp, 'split': False,
    #           # 'key': 'hit_hit',
    #           'key': ('vis_hit', 'con_hit'),
    #           'atlas_name': 'BNA',
    #           'key_vals': (False, True),
    #           'odd_even': False,
    #           'pad_nan': True}

    # kwargs = {'fp': fp, 'split': False,
    #           # 'key': 'hit_hit',
    #           'key': ('inc_vis_hit', 'inc_con_hit'),
    #           'atlas_name': 'BNA',
    #           'key_vals': ('10', '11'),
    #           'odd_even': False,
    #           'pad_nan': True}

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

    results_YA = np.full((len(cols), len(cols)), np.nan)
    results_OA = np.full((len(cols), len(cols)), np.nan)
    for i, p0 in enumerate(partitions):
        if i not in i2name:
            continue
        name0 = i2name[i]
        for j, p1 in enumerate(partitions):
            if j  not in i2name:
                continue
            name1 = i2name[j]
            if j < i:
                continue
            elif i == j:
                print(f'------- {i2name[i]} within conn -------')
                edges = [(a, b) for a in p0 for b in p0 if a < b]
                if not group:
                    print('Activity:')
                    fp = f'cache/p_ss_clf_act_{threshold}_{name0}_{name1}_' \
                         f'{group}' \
                         f'{kwargs["key"]}_{fp}.pkl'
                    f = lambda: partition_classifier(sn_inc_activity, age2idxs,
                                                     edges,
                                                trial_mapper={},
                                                stratification_strategy=2,
                                                activity=True)
                    t_YA_act, t_OA_act = pickle_wrap(fp, f)
                    print(f'{t_YA_act=:.2f}, {t_OA_act=:.2f}')
            else:
                print(f'------- {i2name[i]} x {i2name[j]} -------')
                edges = [(a, b) for a in p0 for b in p1 if a != b]
            fp = f'cache/p_ss_clf_{threshold}_{name0}_{name1}_{group}' \
                 f'{kwargs["key"]}_{fp}.pkl'
            if group:
                # Weirdly givingg so much with respectable accuracy...?
                f = lambda: partition_group_clf(sn_inc_activity, age2idxs, edges,
                                              trial_mapper=trial_mapper,
                                              stratification_strategy=1)
                t_YA, t_OA = pickle_wrap(fp, f)
            else:
                print('Conn:')
                f = lambda: partition_classifier(sn_inc_activity, age2idxs, edges,
                                                  trial_mapper=trial_mapper,
                                                  stratification_strategy=2)
                t_YA, t_OA = pickle_wrap(fp, f)
            print(f'{t_YA=:.2f}, {t_OA=:.2f}')

            print('-')
            results_YA[i2col[i], i2col[j]] = round(t_YA, 3)
            results_OA[i2col[i], i2col[j]] = round(t_OA, 3)
            # ts_YA.append(t_YA)
            # ts_OA.append(t_OA)
    title = f'{kwargs["key"]}, {fp}, {group=}'
    print(f'{kwargs=}')
    print(f'{title=}')
    print('Young:')
    df_YA = pd.DataFrame(results_YA, index=cols, columns=cols)
    print(df_YA)
    print('Old:')
    df_OA = pd.DataFrame(results_OA, index=cols, columns=cols)
    print(df_OA)

if __name__ == '__main__':
    # run_clf()
    # loop_over_ROIs(0.95)
    module_based_clf(group=False)



