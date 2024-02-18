import os

from tqdm import tqdm

from old.modularity import get_partition_matrix
from vendor_partitioning import get_vendor_partitions

os.chdir(r'C:\PycharmProjects_C\SchemeRep')

import numpy as np
import pandas as pd

from atlas_utils import get_atlas
from fMRI_proc import get_ROI_vecs
from organize_bhv import get_trial_info
from utils import timing, stdize, pickle_wrap
from ven_x_dor import get_dfs_conn_trials
import networkx as nx
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

def get_shortest_cc_sn_p(conn_trials_sn_p, sn):
    shortest_l = []
    cc_l = []
    for conn_trial in tqdm(conn_trials_sn_p, desc=f'Doing graph trials: {sn}'):
        conn_trial_ = np.reshape(conn_trial, -1)
        conn_trial_r_ = np.argsort(np.argsort(conn_trial_)).astype(float)
        conn_trial_r_[np.isnan(conn_trial_)] = np.nan
        conn_trial_r = np.reshape(conn_trial_r_, conn_trial.shape)
        conn_trial_r += 0.5
        # print(f'{np.nanmax(conn_trial_r)=}')
        # print(conn_trial_r)
        # conn_trial_r = np.reshape(np.argsort(np.argsort(np.reshape(
        #     conn_trial, -1))), conn_trial.shape) # ranks, NaNs go to highest
        # conn_trial_r_ = np.reshape(conn_trial_r, -1)


        # print(conn_trial_r.shape)
        # print('-')
        # print(np.isnan(conn_trial).shape)
        # quit()
        # conn_trial_r[np.ix_(np.isnan(conn_trial))] = np.nan
        # conn_trial_r += 0.5
        conn_trial_r /= np.nanmax(conn_trial_r) + 0.5
        # print('-'*100)
        # print(conn_trial_r)
        conn_trial_z = stats.norm.ppf(conn_trial_r)
        conn_trial_exp = np.exp(-conn_trial_z)
        # print('-'*100)
        # print(conn_trial_z)
        # quit()


        # conn_trial = np.exp(-conn_trial)
        # print(conn_trial.shape)
        # print(conn_trial)
        # plt.imshow(conn_trial)
        # plt.show()
        # quit()
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
        print(f'{cc=}')
        print(f'{shortest=}')
        quit()
    return shortest_l, cc_l

def apply_df_trial_graph_p(conn_trials_T, p, df_sns_l, key):
    p_trials_T = get_partition_matrix(conn_trials_T, p)
    for conn_trials_sn_p, df_sn in zip(p_trials_T, df_sns_l):
        sn = df_sn['sn'].iloc[0]
        fp_pkl = f'cache/trial_graphs/{sn}_{key}.pkl'
        f = lambda: get_shortest_cc_sn_p(conn_trials_sn_p, sn)
        shortest_l, cc_l = pickle_wrap(fp_pkl, f, easy_override=True)
        df_sn[f'{key}_shortest'] = shortest_l
        df_sn[f'{key}_clustering'] = cc_l
    new_cols = [f'{key}_shortest', f'{key}_clustering']
    return new_cols

@timing
def get_df_trial_graphs(fp='obj7_fMRI', combine_regions=False, anat=True,
                        scrub=False):
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
    new_cols = dd_cols + vv_cols
    # print(f'{new_cols=}')
    df = pd.concat(df_sns_l)
    # print(df[['dd_clustering', 'vv_clustering']])
    # for col in new_cols:
    #
    #     plt.hist(df[col], bins=30)
    #     plt.title(col)
    #     plt.show()

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





