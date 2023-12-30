import numpy as np
from nichord import convert_matrix

from network_clf import generic_prep
from network_funcs import subj_specific_repeated
from subject_specific import conn_ERS_p

def get_edges_mat(edges_l, n):
    edges_mat = np.full((n, n), False)
    for edge in edges_l:
        edges_mat[edge[0], edge[1]] = True
        edges_mat[edge[1], edge[0]] = True
    return edges_mat


def run_network_ERS(threshold=0.95, top_edges_only=False):
    # TODO: look for dm effect via key = con_hit
    fp = ('scn4_fMRI', 'scn4_fMRI')
    fp = ('obj4_fMRI', 'obj4_fMRI')
    # fp = 'obj3_fMRI'
    # fp = 'vis3_fMRI'
    kwargs0 = {'fp': fp, 'split': False,
              'key': 'con_hit',
              'key_vals': (False, True),
              'odd_even': False,
              'fp_all': True}
    kwargs0 = {'fp': fp, 'split': False,
              'key': 'inc',
               'atlas_name': 'BNA',
              'key_vals': (1, 3),
              'odd_even': False,
              'fp_all': True}
    partitions, sn_inc_activity0, age2idxs, top_edges_mat, i2name = \
        generic_prep(kwargs0, threshold=threshold)

    top_edges_l, _ = convert_matrix(top_edges_mat)
    top_edges_set = set(tuple(edge) for edge in top_edges_l)
    kwargs1 = kwargs0.copy()
    kwargs1['fp'] = ('vis3_fMRI', 'con3_fMRI')
    _, sn_inc_activity1, _, _, _ = generic_prep(kwargs1,
                                                threshold=threshold)

    # for i, p in enumerate(partitions):
    #     if i not in i2name:
    #         continue
    #     print(f'------- {i2name[i]} -------')
    #     p_data0 = sn_inc_activity0[..., p, :]
    #     p_data1 = sn_inc_activity1[..., p, :]
    #         conn_ERS_p(sn_inc_activity0, sn_inc_activity1,
    #                    age2idxs, edges)

    for i, p0 in enumerate(partitions):
        if i not in i2name:
            continue
        for j, p1 in enumerate(partitions):
            if j not in i2name:
                continue
            if j < i:
                continue
            elif i == j:
                edges = [(a, b) for a in p0 for b in p0 if a < b]
                print(f'------- {i2name[i]} within activity -------')
                conn_ERS_p(sn_inc_activity0, sn_inc_activity1,
                           age2idxs, edges, activity=True)
                print(f'------- {i2name[i]} within conn -------')
            else:
                edges = [(a, b) for a in p0 for b in p1 if a != b]
                print(f'------- {i2name[i]} x {i2name[j]} -------')

            # edges_mat = edges_matget_edges_mat(edges, sn_inc_activity0.shape[2])
            # print(f'------- {i2name[i]} x {i2name[j]} '
            #       f'variability analysis -------')
            # subj_specific_repeated(sn_inc_activity0, age2idxs, edges_mat,
            #                        None, variability=True)
            # print(f'-*- {i2name[i]} x {i2name[j]} '
            #       f'subject-specific effect analysis -*-')
            # subj_specific_repeated(sn_inc_activity0, age2idxs, edges_mat,
            #                        None, variability=False)
            # continue

            conn_ERS_p(sn_inc_activity0, sn_inc_activity1,
                       age2idxs, edges)
            print(f'-*- top edges only -*-')
            n_edges = len(edges)
            edges = [edge for edge in edges if edge in top_edges_set]
            n_edges_post = len(edges)
            dif = n_edges - n_edges_post
            continue
            print(f'  Edges: {n_edges} - {dif} = {n_edges_post}')
            if n_edges_post < 10:
                continue
            conn_ERS_p(sn_inc_activity0, sn_inc_activity1,
                       age2idxs, edges)



if __name__ == '__main__':
    run_network_ERS()