import numpy as np

from modularity import get_partition_matrix, get_main_partitions
from classifiers import partition_classifier
from network_funcs import load_FC_for_Lifu, reconfiguration, \
    analyze_subject_specific, subj_specific_repeated
from network_clf import generic_prep
from subject_specific import graph_theory
from utils import pickle_wrap


def subject_specific(threshold=0.9):
    fp = 'obj3_fMRI'
    # fp = 'scn3_fMRI'
    # fp = 'bl3_fMRI'
    # kwargs = {'fp': fp, 'split': False,
    #           'key': 'hit_hit',
    #           'key_vals': (True, False),
    #           'odd_even': True}

    kwargs = {'fp': fp, 'split': False,
              'key': 'inc',
              'key_vals': (1, 3),
              'odd_even': True}

    sn_inc_conn, sn_conn, age2idxs, sn_inc_activity = pickle_wrap(None,
                                                                  load_FC_for_Lifu, kwargs=kwargs,
                                                                  verbose=1, easy_override=False,
                                                                  cache_dir='../cache')
    fp = 'cache/test.pkl'
    partitions, top_edges_mat = pickle_wrap(fp, lambda: get_main_partitions(
        sn_conn, plot=False, threshold=threshold), easy_override=False)
    # partitions, top_edges_mat = get_main_partitions(sn_conn, plot=False,
    #                                                 threshold=threshold)
    # partitions = [[0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 50, 51, 62, 164, 166, 167, 176, 177, 178, 179, 186, 187, 232], [48, 49, 68, 69, 76, 77, 78, 80, 82, 83, 86, 87, 88, 89, 92, 93, 94, 95, 102, 103, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 165, 210, 211, 212, 213, 214, 215, 216, 217], [6, 7, 8, 9, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 138, 139, 147, 148, 149, 154, 155, 158, 159, 160, 161, 182, 183, 184], [81, 84, 85, 90, 91, 96, 97, 98, 99, 100, 101, 104, 105, 106, 107, 134, 135, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209], [218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245], [60, 61, 70, 71, 72, 73, 74, 75, 79, 120, 121, 122, 123, 144, 145, 156, 157, 162, 163, 168, 169, 170, 171, 172, 173], [136, 137, 140, 141, 142, 143, 146, 150, 151, 152, 153, 174, 175, 180, 181, 185]]

    # partitions = [list(range(246)) for i in range(2)]

    sn_inc_conn[..., ~top_edges_mat] = np.nan
    i2name = {0.9: {0: 'PFC+', 1: 'MTL+'},
              0.95: {0: 'MTL+', 2: 'dPFC', 5: 'vPFC'}}

    for i, p in enumerate(partitions):
        if i not in i2name[threshold]:
            continue
        print('-' * 50)
        name = i2name[threshold][i]
        print(f'Partition: {name} ({i})')
        p_mat = get_partition_matrix(sn_inc_conn, p)
        analyze_subject_specific(p_mat, age2idxs)


def INC_reconfig(threshold=0.9):
    fp = 'obj3_fMRI'
    kwargs = {'fp': fp, 'split': False,
              'key': 'inc',
              # 'key': 'rand',
              'key_vals': (1, 2, 3)}
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


    for i, p in enumerate(partitions):
        if i not in i2name[threshold]:
            continue
        print('-' * 50)
        name = i2name[threshold][i]
        print(f'Partition: {name} ({i})')
        M0, M1, d, dif = reconfiguration(sn_inc_conn, age2idxs, p,
                                         title=f'{name}, Inc vs Neu',
                                         col0=0, col1=1)
        M0, M1, d, dif = reconfiguration(sn_inc_conn, age2idxs, p,
                                         title=f'{name}, Inc vs Con',
                                         col0=0, col1=2)
        M0, M1, d, dif = reconfiguration(sn_inc_conn, age2idxs, p,
                                         title=f'{name}, Neu vs Con',
                                         col0=1, col1=2)



def rand_test(threshold=0.9, way3=False):
    difs = []
    ds = []
    for nsim in range(100):
        fp = 'obj3_fMRI'
        kwargs = {'fp': fp, 'split': False,
                  # 'key': 'inc',
                  'key': 'rand',
                  'key_vals': (1, 2, 3) if way3 else (False, True)}
        sn_inc_conn, sn_conn, age2idxs, sn_inc_activity = pickle_wrap(None,
                                                                      load_FC_for_Lifu, kwargs=kwargs,
                                                                      verbose=1, easy_override=True,
                                                                      cache_dir='../cache')
        partitions, top_edges_mat = get_main_partitions(sn_conn, plot=False,
                                                        threshold=threshold)
        sn_inc_conn[:, :, ~top_edges_mat] = np.nan
        i2name = {0.9: {1: 'MTL+'}}
        ps_good = [partitions[i] for i in i2name[threshold]]

        for i, p in enumerate(partitions):
            if i not in i2name[threshold]:
                continue
            if len(p) < 5:
                continue
            print('-' * 50)
            name = i2name[threshold][i]
            print(f'Partition: {name} ({i})')
            M0, M1, d, dif = reconfiguration(sn_inc_conn, age2idxs, p,
                                             title=f'{name}, Inc vs Neu',
                                             col0=0, col1=1, plot=False)
            ds.append(d)
            difs.append(dif)
            if way3:
                M0, M1, d, dif = reconfiguration(sn_inc_conn, age2idxs, p,
                                                 title=f'{name}, Inc vs Con',
                                                 col0=0, col1=2, plot=False)
                ds.append(d)
                difs.append(dif)
                M0, M1, d, dif = reconfiguration(sn_inc_conn, age2idxs, p,
                                                 title=f'{name}, Neu vs Con',
                                                 col0=1, col1=2, plot=False)
                ds.append(d)
                difs.append(dif)
            print(f'{len(ds)}')
            print(f'{np.mean(ds)=} [{np.std(ds)=})]')
            print(f'{np.mean(difs)=} [{np.std(difs)=})]')


def variability(threshold=0.99):

    fp = 'obj4_fMRI'
    kwargs = {'fp': fp, 'split': False,
              'key': 'inc',
              'atlas_name': 'schaefer',
              'key_vals': (1, 3),
              'odd_even': False,
              }

    fp = ('con3_fMRI', 'vis3_fMRI')
    kwargs = {'fp': fp, 'split': False,
              'key': 'hit_hit',
              # 'key': 'con_hit',
              'atlas_name': 'schaefer',
              'key_vals': (False, True),
              'odd_even': False,
              }

    # sn_inc_conn, sn_conn, age2idxs, sn_inc_activity = pickle_wrap(None,
    #                                                               load_FC_for_Lifu, kwargs=kwargs,
    #                                                               verbose=1, easy_override=False,
    #                                                               cache_dir='../cache')
    # sn_inc_activity = np.array(sn_inc_activity)
    #
    # # partitions, top_edges_mat = pickle_wrap(fp, lambda: get_main_partitions(
    # #     sn_conn, plot=True, threshold=threshold, overlapping=True),
    # #                                         easy_override=True)
    #
    # fp = 'cache/test.pkl'
    # partitions, top_edges_mat = pickle_wrap(fp, lambda: get_main_partitions(
    #     sn_conn, plot=False, threshold=threshold), easy_override=False)
    #
    # sn_inc_conn_unthresh = sn_inc_conn.copy()
    # sn_inc_conn[:, :, ~top_edges_mat] = np.nan
    #
    # # partitions = [list(range(246))]
    #
    # i2name = {0.9: {0: 'PFC+', 1: 'MTL+'},
    #           0.95: {0: 'MTL+', 2: 'dPFC', 5: 'vPFC'}}
    # i2name = {0.9: {0: 'PFC+', 1: 'MTL+',  3: 'Vis'}}

    partitions, sn_inc_activity, age2idxs, top_edges_mat, i2name = \
        generic_prep(kwargs, threshold=threshold)

    for i, p in enumerate(partitions):
        # if i < 1: continue
        if i not in i2name:
            continue
        p_top_edges = get_partition_matrix(top_edges_mat, p)
        p_data = sn_inc_activity[..., p, :]
        # print(f'{p=}')
        print(f'------- {i2name[i]} -------')

        # print(' Subject-specific:')
        # continue
        graph_theory(sn_inc_activity, age2idxs, p_top_edges, p,
                     measure='shortest')
        # # print('-*-*-')
        graph_theory(sn_inc_activity, age2idxs, p_top_edges, p,
                     measure='clustering')
        # # print('-*-*-')
        # # continue
        # graph_theory(sn_inc_activity, age2idxs, p_top_edges, p,
        #              measure='omega')
        # print('-*-*-')
        # graph_theory(sn_inc_activity, age2idxs, p_top_edges, p)


        # subj_specific_repeated(p_data, age2idxs, p_top_edges, p,
        #                        variability=False)
        # print(' Variability:')
        # subj_specific_repeated(p_data, age2idxs, p_top_edges, p,
        #                        variability=True)
        # print()




if __name__ == '__main__':
    # THRESHOLD = 0.9
    # Fig15_Table2_analyses(THRESHOLD)
    # quit()
    # Fig16a_analyses(THRESHOLD, memory_type='vis_hit')
    # quit()
    # Fig16a_analyses(THRESHOLD, memory_type='con_hit')
    # Fig16a_analyses(THRESHOLD, memory_type='hit_hit')

    # Fig17_analyses(THRESHOLD)
    # INC_reconfig(THRESHOLD)
    # rand_test(THRESHOLD)
    # subject_specific()
    variability()
