import os

from tqdm import tqdm

from old.modularity import get_partition_matrix, get_partition_cross
from vendor_partitioning import get_vendor_partitions

os.chdir(r'/')

import numpy as np
import pandas as pd

from atlas_utils import get_atlas
from fMRI_proc import get_ROI_vecs
from organize_bhv import get_trial_info
from utils import timing, stdize, pickle_wrap
from load_more import get_dfs_conn_trials
# import networkx as nx
import matplotlib.pyplot as plt
import scipy.stats as stats

def get_graph(adj, tile=.95):
    adj = adj.copy()
    thresh = np.nanquantile(adj, tile)
    # adj[adj < thresh] = 0
    # adj[adj >= thresh] = 1
    G = nx.from_numpy_array(adj)
    return G

def get_graph_scores(G):
    # score = nx.average_shortest_path_length(G)
    # score_d = nx.shortest_path_length(G)
    # worst_M_score = 0
    # for node, d in score_d:
    #     node_score = np.mean(list(d.values()))
    #     if node_score > worst_M_score:
    #         worst_M_score = node_score
    # if len(biggest) < biggest_of_any:
    #     missing = biggest_of_any - len(biggest)
    #     score_new = (worst_M_score * missing +
    #                  score * len(biggest)) / biggest_of_any
    #     # print(f'{score} | {score_new}')
    #     score = score_new
    #
    # return score
    comp = nx.algorithms.components.connected_components(G)
    biggest = set()
    for c in comp:
        if len(c) > len(biggest):
            biggest = c

    print(f'{biggest=}')
    return

    # if len(biggest) < 20 and measure != 'len':
        # print(f'Bad: {age=}, {sn=}, {inc0=}, {len(biggest)=}')
        # break
    if len(biggest) > biggest_of_any:
        biggest_of_any = len(biggest)
        # print(f'Biggest: {age=}, {sn=}, {inc0=}, {len(biggest)=}')
    G = G.subgraph(biggest)
    # print('Calculating small worldness')
    # smol = nx.sigma(G, niter=10, nrand=2)
    if measure == 'len':
        score = len(biggest)
    elif measure == 'shortest':
        score = nx.average_shortest_path_length(G)
        if len(biggest) < biggest_of_any: # TODO: better
            score_d = nx.shortest_path_length(G)
            worst_M_score = 0
            for node, d in score_d:
                node_score = np.mean(list(d.values()))
                if node_score > worst_M_score:
                    worst_M_score = node_score
            if len(biggest) < biggest_of_any:
                missing = biggest_of_any - len(biggest)
                score_new = (worst_M_score * missing +
                             score * len(biggest)) / biggest_of_any
                # print(f'{score} | {score_new}')
                score = score_new
    elif measure == 'clustering':
        score = nx.average_clustering(G)
    elif measure == 'closeness_centrality':
        score = nx.closeness_centrality(G)
        print(f'Closeness: {age=}, {sn=}, {inc0=} | {score=:.3f}')
    elif measure == 'sigma':
        score = nx.sigma(G, niter=10, nrand=5, seed=0)
        print(f'Sigma: {age=}, {sn=}, {inc0=} | {score=:.3f}')
    elif measure == 'omega':
        score = nx.omega(G, niter=10, nrand=5, seed=0)
        print(f'Omega: {age=}, {sn=}, {inc0=} | {score=:.3f}')

    else:
        raise NotImplementedError

def get_shortest_cc_sn_p(conn_trials_sn_p, sn, rank_std=True):
    shortest_l = []
    cc_l = []

    if rank_std:
        conn_trials_sn_p_ = np.reshape(conn_trials_sn_p,
                                       (conn_trials_sn_p.shape[0], -1))
        conn_trials_sn_p_r_ = np.argsort(np.argsort(conn_trials_sn_p_))
        conn_trials_sn_p_r = np.reshape(conn_trials_sn_p_r_,
                                        conn_trials_sn_p.shape)
        conn_trials_sn_p_r = conn_trials_sn_p_r.astype(float)
        conn_trials_sn_p_r[np.isnan(conn_trials_sn_p)] = np.nan
        conn_trials_sn_p_r += 0.5
        conn_trials_sn_p_r /= np.nanmax(conn_trials_sn_p_r) + 0.5
        conn_trials_sn_p_z = stats.norm.ppf(conn_trials_sn_p_r)
        # M = np.nanmean(conn_trials_sn_p_z)
        # print(f'{M=}')
        # print('toast')
        # quit()
    else:
        conn_trials_sn_p_z = conn_trials_sn_p
    conn_trials_sn_p_exp = np.exp(-conn_trials_sn_p_z)

    for conn_trial_exp in tqdm(conn_trials_sn_p_exp,
                           desc=f'Doing graph trials: {sn}'):
        G = nx.from_numpy_array(conn_trial_exp)
        try:
            shortest = nx.average_shortest_path_length(G, weight='weight')
            cc = nx.average_clustering(G, weight='weight')
        except ValueError as e:
            print(f'Bad networkx: {e=}')
            shortest = np.nan
            cc = np.nan
        shortest_l.append(shortest)
        cc_l.append(cc)
        # print(f'{cc=}')
        # print(f'{shortest=}')
        # quit()
    return shortest_l, cc_l

def apply_df_trial_graph_p(conn_trials_T, p, df_sns_l, key, p1=None,
                           rank_std=True):
    if p1 is None:
        p_trials_T = get_partition_matrix(conn_trials_T, p)
    else:
        p_trials_T = get_partition_cross(conn_trials_T, p, p1)
    for conn_trials_sn_p, df_sn in zip(p_trials_T, df_sns_l):
        sn = df_sn['sn'].iloc[0]
        rank_std_str = '_rank_std' if rank_std else ''

        p_idxs_str = ''.join([str(p_i) for p_i in p])
        num_char = len(p_idxs_str)
        target_char = 20
        if target_char < num_char:
            skip = num_char // target_char
            p_idxs_str = p_idxs_str[::skip]
        fp_pkl = f'cache/trial_graphs/{sn}_{p_idxs_str}{rank_std_str}.pkl'
        # fp_pkl = f'cache/trial_graphs/{sn}_{key}{rank_std_str}.pkl'
        f = lambda: get_shortest_cc_sn_p(conn_trials_sn_p, sn,
                                         rank_std=rank_std)
        shortest_l, cc_l = pickle_wrap(f, fp_pkl, easy_override=False)
        # kwargs = {'conn_trials_sn_p': conn_trials_sn_p,
        #           'sn': sn, 'rank_std': True}
        # shortest_l, cc_l = pickle_wrap(None, get_shortest_cc_sn_p,
        #                                kwargs=kwargs,
        #                                easy_override=True)
        df_sn[f'{key}_shortest'] = shortest_l
        df_sn[f'{key}_clustering'] = cc_l
    new_cols = [f'{key}_shortest', f'{key}_clustering']
    return new_cols

@timing
def get_df_trial_graphs(fp='obj7_fMRI', anat=True, scrub=False):
    # conn_trials, df_sns_l = pickle_wrap(None, get_dfs_conn_trials,
    #                                     kwargs={'fp': fp,
    #                                             'single': True},
    #                                     easy_override=True)
    conn_trials, df_sns_l = get_dfs_conn_trials(fp)
    conn_trials = np.nanmean(conn_trials, axis=1)

    p_dorsal, p_ventral, p_d_ant, p_d_pos, p_v_ant, p_v_pos, matrix_mask = \
        get_vendor_partitions(age='healthy', flip=True, anat=anat, scrub=scrub)

    conn_trials_T = conn_trials.transpose((0, 3, 1, 2))

    num_nodes = conn_trials_T.shape[-1]
    diag_idxs = np.diag_indices(num_nodes)
    conn_trials_T[:, :, diag_idxs[0], diag_idxs[1]] = np.nan

    dd_cols = apply_df_trial_graph_p(conn_trials_T, p_dorsal, df_sns_l, 'dd')
    vv_cols = apply_df_trial_graph_p(conn_trials_T, p_ventral, df_sns_l, 'vv')
    # dd_cols = apply_df_trial_graph_p(conn_trials_T, p_d_pos, df_sns_l, 'dd',
    #                                  p1=p_d_ant)
    # vv_cols = apply_df_trial_graph_p(conn_trials_T, p_v_pos, df_sns_l, 'vv',
    #                                  p1=p_v_ant)
    p_dv_ant = p_d_ant + p_v_ant
    p_dv_pos = p_d_pos + p_v_pos
    dv_ant_cols = apply_df_trial_graph_p(conn_trials_T, p_dv_ant, df_sns_l,
                                         'dv_ant')
    dv_pos_cols = apply_df_trial_graph_p(conn_trials_T, p_dv_pos, df_sns_l,
                                         'dv_pos')
    new_cols = dd_cols + vv_cols +  dv_ant_cols + dv_pos_cols
    df = pd.concat(df_sns_l)
    return df, new_cols



if __name__ == '__main__':
    test = np.array([[0.5, np.nan, 2.4], [1.1, 13.1, np.nan]])
    # print(np.argsort(np.argsort(np.reshape(test, -1))))
    # print(np.reshape(test, -1))
    test_out = np.reshape(np.argsort(np.argsort(np.reshape(test, -1))),
                          test.shape) # argsort^2 = rank
    print(test_out.shape)
    print(test_out)
    # print(np.argsort(test, axis=None))
    quit()

    get_df_trial_graphs()





