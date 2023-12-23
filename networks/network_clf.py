import warnings

import numpy as np
from connsearch.report import plot_ROI_scores
from tqdm import tqdm

from atlas_utils import get_atlas
from modularity import get_partition_matrix, get_main_partitions
from classifiers import partition_classifier
from network_funcs import load_FC_for_Lifu, reconfiguration, \
    analyze_subject_specific, subj_specific_repeated
from utils import pickle_wrap

warnings.filterwarnings('ignore',
                        message='invalid value encountered in divide')


def generic_prep(kwargs, threshold=0.9):
    sn_inc_conn, sn_conn, age2idxs, sn_inc_activity = pickle_wrap(None,
                                                                  load_FC_for_Lifu, kwargs=kwargs,
                                                                  verbose=1, easy_override=False,
                                                                  cache_dir='../cache')
    sn_inc_activity = np.array(sn_inc_activity)
    fp = 'cache/obj4_partitions_test.pkl'
    partitions, top_edges_mat = pickle_wrap(fp, lambda: get_main_partitions(
        sn_conn, plot=True, threshold=threshold), easy_override=False)
    sn_inc_conn_unthresh = sn_inc_conn.copy()
    sn_inc_conn[:, :, ~top_edges_mat] = np.nan
    i2name = {0.9: {0: 'PFC', 1: 'SM', 2: 'Vis', 3: 'MTL',
                    4: 'pSM', 5: 'PCun', 6: 'Limbic'}}
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


    kwargs = {'fp': fp, 'split': False,
              'key': 'inc',
              'key_vals': (1, 3),
              'odd_even': False,
              }
    kwargs = {'fp': fp, 'split': False,
              'key': 'inc_match14_strict',
              'key_vals': (False, True),
              'odd_even': False,
              }
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
                                          trial_mapper=trial_mapper)
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

def module_based_clf(threshold=0.9):
    fp = 'obj4_fMRI'
    # fp = 'vis3_fMRI'

    kwargs = {'fp': fp, 'split': False,
              'key': 'inc',
              'key_vals': (1, 3),
              'odd_even': False,
              }

    kwargs = {'fp': fp, 'split': False,
              'key': 'inc_match',
              'key_vals': (False, True),
              'odd_even': False,
              }
    # kwargs = {'fp': fp, 'split': False,
    #           'key': 'inc_match14_strict',
    #           'key_vals': (False, True),
    #           'odd_even': False,
    #           }

    # kwargs = {'fp': fp, 'split': False,
    #           # 'key': 'hit_hit',
    #           'key': 'vis_hit',
    #           'key_vals': (False, True),
    #           'odd_even': False,
    #           'pad_nan': True}

    partitions, sn_inc_activity, age2idxs, top_edges_mat, i2name = \
        generic_prep(kwargs, threshold=threshold)
    age2idxs[3] = age2idxs[1] + age2idxs[2]
    trial_mapper = {}
    for i, p0 in enumerate(partitions):
        name0 = i2name[i]
        for j, p1 in enumerate(partitions):
            name1 = i2name[j]
            if j > i:
                continue
            elif i == j:
                edges = [(a, b) for a in p0 for b in p0 if a < b]
            else:
                edges = [(a, b) for a in p0 for b in p1 if a != b]
            print('-')
            t_YA, t_OA = partition_classifier(sn_inc_activity, age2idxs, edges,
                                              trial_mapper=trial_mapper,
                                              stratification_strategy=1)
            print(f'{name0} x {name1}: {t_YA=:.2f}, {t_OA=:.2f}')

            # ts_YA.append(t_YA)
            # ts_OA.append(t_OA)

if __name__ == '__main__':
    # run_clf()
    # loop_over_ROIs(0.9)
    module_based_clf()



