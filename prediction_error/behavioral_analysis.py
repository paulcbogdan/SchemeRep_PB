import pandas as pd

from organize_bhv import get_trial_info
from org_sns import get_all_sns
from collections import defaultdict

from stim import get_semantic_vectors, get_DNN_vecs
import numpy as np
from pprint import pprint

import matplotlib.pyplot as plt
import matplotlib
import scipy.stats as stats

def stdize_vecs(d_vecs):
    vecs_all = []
    for img, vec in d_vecs.items():
        vecs_all.append(vec)
    vecs_all = np.array(vecs_all)
    M = np.mean(vecs_all, axis=0)
    SD = np.std(vecs_all, axis=0)
    for img, vec in d_vecs.items():
        d_vecs[img] = (vec - M) / SD


def test_vectors(semantic=True, DNN_layer=4):
    age2sn = get_all_sns(ret=False)
    pairs_all = []
    inc2pairs = defaultdict(set)
    inc2resps = defaultdict(list)
    for sn in age2sn[1]:
        df_sn = get_trial_info(sn, easy_override=False)
        for inc, obj, scn, resp in zip(df_sn['inc'], df_sn['obj'],
                                       df_sn['scene'], df_sn['per_con']):
            inc2pairs[inc].add((obj, scn))
            inc2resps[inc].append(resp)
    # pprint(cin2pairs)
    # for inc, pairs in inc2pairs.items():
    #     print(f'{inc}: {pairs}')
    #     print(f'{inc}: {np.nanmean(inc2resps[inc])}')
    #     print()
    #     print()
    # quit()

    if semantic:
        d_vecs = get_semantic_vectors()
    else:
        d_vecs = get_DNN_vecs(DNN_layer=DNN_layer, PCA=True)

    # stdize_vecs(d_vecs)

    inc2dif_l = defaultdict(list)
    Ms = []
    SDs = []
    SEs = []
    # for inc, pairs in inc2pairs.items():
    obj2inc2dif = defaultdict(dict)
    for inc in [1, 2, 3]:
        pairs = inc2pairs[inc]
        for pair in pairs:
            vec_obj = d_vecs[pair[0]]
            vec_scn = d_vecs[pair[1]]
            dif = abs(vec_obj - vec_scn)
            m = np.mean(dif)
            inc2dif_l[inc].append(m)
            obj2inc2dif[pair[0]][inc] = m
        M = np.mean(inc2dif_l[inc])
        SD = np.std(inc2dif_l[inc])
        SE = SD / np.sqrt(len(inc2dif_l[inc]))
        print(f'{inc}: {M=:.4f} [{SD=:.4f}, {SE=:.4f}]')
        Ms.append(M)
        SDs.append(SD)
        SEs.append(SE)

    difs = [d[3] - d[1] for obj, d in obj2inc2dif.items()]
    M_dif = np.mean(difs)
    SD_dif = np.std(difs)
    SE_dif = SD_dif / np.sqrt(len(difs))
    t = M_dif / SE_dif
    p = stats.t.cdf(t, len(difs) - 1)
    print(f'{M_dif=:.4f} [{SE_dif=:.4f}], {t=:.3f}, {p=:.3f}')
    quit()


    font = {'size': 14}
    matplotlib.rc('font', **font)
    # plt.xticks([0, 1, 2], ['Incongruent', 'Neutral', 'Congruent'])
    plt.bar(['Incongruent', 'Neutral', 'Congruent'], Ms, yerr=SEs,
            color=['red', 'purple', 'blue'])
    plt.ylabel('Mean of abs(obj - scene) vector')
    # plt.yticks([0, 0.05, 0.1, 0.15, 0.2])
    # plt.ylim(104, 110)
    # plt.title('Perceptual')
    plt.ylim(0.14, 0.19)
    plt.title('Semantic')

    plt.tight_layout()
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # plt.errorbar([0, 1, 2], Ms, yerr=SDs, fmt='o')
    plt.show()

def test_U_memory(DV='con_hit'):
    low_acc_sns = {'104', '109', '115', '119'}
    age2sn = get_all_sns('obj4_fMRI')
    df_l = []
    for sn in age2sn[1]:
        if sn in low_acc_sns:
            continue
        df_sn = get_trial_info(sn, easy_override=False)
        # df_sn['inc'] = df_sn['per_inc']
        df_l.append(df_sn)
        # is_in = 'con_hit' in df_sn.columns
        # print(is_in)
        # print(df_sn.columns)

        # print(df_sn['con_hit'])
        # quit()
    df = pd.concat(df_l)
    # print(df['con_hit'].value_counts(dropna=False))
    # print(df['hit_hit'].value_counts(dropna=False))

    print(list(df.columns))
    dv2name = {'con_hit': 'Conceptual hit',
               'vis_hit': 'Visual hit',
               'hit_hit': 'Both hit'}

    df_M = df.groupby(['sn', 'inc'])[DV].mean()
    df_SE = df_M.groupby(['inc']).sem()
    df_M = df_M.groupby(['inc']).mean()
    key2name = {1: 'Incongruent',
                2: 'Neutral',
                3: 'Congruent'}
    Ms = []
    SEs = []
    incs = []
    for inc in [1, 2, 3]:
        print(f'{key2name[inc]}: {df_M[inc]:.4f} [{df_SE[inc]:.4f}]')
        Ms.append(df_M[inc])
        SEs.append(df_SE[inc])
        incs.append(key2name[inc])
    font = {'size': 14}
    matplotlib.rc('font', **font)
    plt.bar(incs, Ms, yerr=SEs, color=['red', 'purple', 'blue'])
    plt.ylabel(dv2name[DV])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()
    print('-'*100)


    df_M = df.groupby(['sn', 'per_inc'])[DV].mean()
    df_SE = df_M.groupby(['per_inc']).sem()
    df_M = df_M.groupby(['per_inc']).mean()
    key2name = {1: '1\n(Perceived\nincongruent)',
                2: '2',
                3: '3',
                4: '4\n(Perceived\ncongruent)'}
    Ms = []
    SEs = []
    incs = []
    for inc in [1, 2, 3, 4]:
        print(f'{key2name[inc]}: {df_M[inc]:.4f} [{df_SE[inc]:.4f}]')
        Ms.append(df_M[inc])
        SEs.append(df_SE[inc])
        incs.append(key2name[inc])
    plt.bar(incs, Ms, yerr=SEs, color=['red',
                                       'mediumvioletred',
                                       'mediumslateblue',
                                       'blue'])
    plt.ylabel(dv2name[DV])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # test_vectors()
    # quit()
    for dv in ['con_hit', 'vis_hit', 'hit_hit']:
        test_U_memory(dv)


