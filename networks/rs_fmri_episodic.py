import os

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from atlas_utils import get_atlas
from fMRI_proc import get_ROI_vecs
from old_Apr6.fluctuations import get_df_networks
from org_sns import get_sns
from organize_bhv import get_trial_info
from utils import pickle_wrap, stdize
import scipy.stats as stats

os.chdir(r'H:\PycharmProjects_H\SchemeRep')
from load_more import load_a, get_sn_rs


def get_sn_retrieval_template(sess='con', fp_fMRI_col='con7_fMRI',
                              ROIs_analyze=('Hipp',)):
    age2sn = get_sns(sess)
    sns = age2sn[1] + age2sn[2]
    bad_rs_sns = {'133', '239'} # todo add 239 maybe
    sns = [sn for sn in sns if sn not in bad_rs_sns]
    atlas = get_atlas(combine_regions=False)
    ROIs = atlas['ROIs']
    ROI_nums = atlas['ROI_nums']
    ROI_regions = atlas['ROI_regions']

    difs = []
    sn2dif = {}
    for sn in tqdm(sns, desc='getting templates'):
        df_sn = get_trial_info(sn, only_one=sess, easy_override=True)
        # print(df_sn)
        # quit()
        ROI2vecs = get_ROI_vecs(sn, atlas, fp_fMRI_col, df_sn,
                                nan_thresh=1.01,
                                drop_nan_voxels=False,
                                org_by_region=False,
                                easy_override=False,
                                combine_regions=False)

        # cnt = df_sn[['old_new', 'con_hit', ]].value_counts(dropna=False)
        # print(cnt)
        # continue
        # quit()
        key = f'{sess}_hit'
        key_resp = f'{sess}_resp'
        if sess == 'vis':
            df_sn['old_new'] = df_sn['vis_type']
            cnt = df_sn[['old_new', 'vis_resp', ]].value_counts(dropna=False)
            # print(cnt)
            true_hit = (df_sn['old_new'] == 'old') & (df_sn['vis_resp'] == 'old')
            new_CR = ~true_hit
        else:
            true_hit = (df_sn['old_new'] == 'old') & (df_sn[key])
            new_CR = ~true_hit

            # true_hit = (df_sn['old_new'] == 'old') & (df_sn[key_resp] == 4)

            true_miss = (df_sn['old_new'] == 'old') & (df_sn[key] == False) # excl NaN
            new_FA = (df_sn['old_new'] == 'new') & (df_sn[key])
            # print(df_sn[['old_new', key]].value_counts(dropna=False))
            # new_CR = (df_sn['old_new'] == 'new') & (df_sn[key] == False)

        df_sn_hit = df_sn[true_hit]
        # print(len(df_sn_hit))
        df_sn_not_hit = df_sn[new_CR]
        # print(len(df_sn_not_hit))
        # quit()
        assert len(df_sn_hit) > 0, f'{sn=}, {sess=}, {len(df_sn_hit)=}'
        assert len(df_sn_not_hit) > 0, f'{sn=}, {sess=}, {len(df_sn_not_hit)=}'
        # df_sn_hit = df_sn[df_sn[f'{sess}_hit'] == 1]
        # df_sn_not_hit = df_sn[df_sn[f'{sess}_hit'] != 1]


        target_ROIs = []
        for ROI in ROIs:
            for ROI_analyze in ROIs_analyze:
                if ROI_analyze in ROI:
                    target_ROIs.append(ROI)
                    break
            # if 'Hipp' in ROI:
            #     target_ROIs.append(ROI)
        # target_ROIs = [ROI for ROI in ROIs if 'Hipp' in ROI]
        vecs_all = [ROI2vecs[ROI] for ROI in target_ROIs]
        vecs_all = np.concatenate(vecs_all, axis=1)
        vecs_all_hit = vecs_all[df_sn_hit.index]
        vecs_all_miss = vecs_all[df_sn_not_hit.index]

        hit_response = np.nanmean(vecs_all_hit, axis=0)
        miss_response = np.nanmean(vecs_all_miss, axis=0)

        hit_SD = np.std(vecs_all_hit, axis=0)
        miss_SD = np.std(vecs_all_miss, axis=0)

        combined_SD = np.sqrt(hit_SD**2 + miss_SD**2)

        dif = (hit_response - miss_response) / combined_SD
        sn2dif[sn] = dif
        difs.append(dif)
        # print(f'{dif.shape=}')
        # print(dif)
        # quit()
    # quit()

    return np.array(difs), sns


def get_rs_hc(norm_std=False, YA_only=False,
              sns_key='con', compcor=True,
              clean=True, medium=True, ROIs_analyze=('Hipp',)):
    age2sn = get_sns(sns_key)

    if YA_only:
        sns = age2sn[1]
    else:
        sns = age2sn[1] + age2sn[2]

    atlas = get_atlas(combine_regions=False)
    bad_rs_sns = {'133', '239'} # todo add 239 maybe
    sns = [sn for sn in sns if sn not in bad_rs_sns]
    ROIs = atlas['ROIs']
    ROI_nums = atlas['ROI_nums']
    ROI_regions = atlas['ROI_regions']


    # sn2region_vecs = {}
    region_vecs_all = []
    for i, sn in tqdm(enumerate(sns), desc='Loading fMRI'):
        data = pickle_wrap(get_sn_rs, kwargs={'sn': sn, 'clean': clean,
                                              'compcor': compcor,
                                              'light': False,
                                              'medium': medium,
                                              'trad': False,
                                              'near_OG': False,
                                              'true_OG': False},
                           easy_override=False)
        atlas_roi = np.zeros_like(data[0])
        region_vecs_hc = []
        for j, (ROI, ROI_num, region) in enumerate(
                zip(ROIs, ROI_nums, ROI_regions)):
            good = False
            for ROI_analyze in ROIs_analyze:
                if ROI_analyze in ROI:
                    good = True
                    break
            # print(f'{ROI=} | {good=}')
            if not good:
                continue
            # quit()

            # if region != 'Hipp':
            #     continue
            #     atlas_roi += atlas['maps'].get_fdata() == ROI_num
            # region_vecs = data[atlas_roi]
            atlas_roi = atlas['maps'].get_fdata() == ROI_num
            region_vecs = data[atlas_roi]
            region_vecs_hc.append(region_vecs)

        region_vecs_hc = np.concatenate(region_vecs_hc).T
        region_vecs_all.append(region_vecs_hc)
        # sn2region_vecs[sn] = region_vecs_hc
    region_vecs_all = np.array(region_vecs_all)
    return region_vecs_all, sns

def get_episodic_scores(sns_key='con', YA_only=False,
                        ROIs_analyze=('Hipp',), group_avg=False):
    age2sn = get_sns(sns_key)

    if YA_only:
        sns = age2sn[1]
    else:
        sns = age2sn[1] + age2sn[2]
    bad_rs_sns = {'133', '239'} # todo add 239 maybe
    sns = [sn for sn in sns if sn not in bad_rs_sns]


    kw = {'ROIs_analyze': ROIs_analyze, 'sess': sns_key,
          'fp_fMRI_col': f'{sns_key}7_fMRI'}
    difs, sns1 = pickle_wrap(get_sn_retrieval_template, kwargs=kw,
                             easy_override=False)
    difs = stdize(difs, axis=1, nans=True)

    if group_avg:
        difs = np.nanmean(difs, axis=0)[None, :]

    # scores = scores[np.abs(scores) > 2]
    # print(difs.shape)

    # upper_quant = np.nanquantile(np.reshape(difs, -1), 0.99, axis=0)
    upper_quant = np.nanquantile(difs, 0.99, axis=0)

    # lower_quant = np.nanquantile(np.reshape(difs, -1), 0.01, axis=0)
    lower_quant = np.nanquantile(difs, 0.01, axis=0)

    # print(f'{upper_quant=}, {lower_quant=}')
    difs[(difs < upper_quant) & (difs > lower_quant)] = np.nan
    difs[difs > 0] = 1
    difs[difs < 0] = -1
    difs = stdize(difs, axis=1, nans=True) # there'll be roughly equal number on each side
    # difs[np.isnan(difs)] = 0
    # difs[np.abs(difs) < 2] = 0
    # print(difs)
    # plt.imshow(difs, aspect='auto', interpolation='none')
    # plt.colorbar()
    # plt.show()

    # above0 = np.sum(difs > 0)
    # below0 = np.sum(difs < 0)
    # # print(f'{above0=}, {below0=}')
    # assert above0 == below0, f'{above0=}, {below0=}'
    # quit()
    # difs[np.abs(difs) < 3] = np.nan


    # rnd_normal = np.random.normal(size=np.reshape(difs, -1).shape)
    # plt.hist(rnd_normal, bins=100, range=(-5, 5))
    # plt.show()

    # plt.hist(np.reshape(difs, -1), bins=100, range=(-5, 5))
    # plt.show()
    # quit()

    kw = {'ROIs_analyze': ROIs_analyze, 'sns_key': sns_key,}
    rs_vecs, sns0 = pickle_wrap(get_rs_hc, kwargs=kw, easy_override=False)
    assert tuple(sns0) == tuple(sns1)

    scores = rs_vecs * difs[:, None, :]
    scores = np.nanmean(scores, axis=2)
    scores = stdize(scores, axis=-1)


    sn2scores = {sn: scores[i] for i, sn in enumerate(sns)}
    return scores, sn2scores
    # print(f'{scores.shape=}')




def do_episodic_x_vd(fp='rs_medium', ROIs_analyze=('Hipp',),
                     mem='con'):
    df, _ = pickle_wrap(get_df_networks, kwargs={'fp': fp,
                                                 'anat_ver': 3,
                                                 'add_hemi': False},
                        easy_override=False)
    kw = {'sns_key': mem, 'YA_only': False, 'ROIs_analyze': ROIs_analyze,
          'group_avg': False}
    scores, sn2scores = pickle_wrap(get_episodic_scores, kwargs=kw,
                                    easy_override=False)
    # scores, sn2scores = get_episodic_scores(sns_key=mem, YA_only=False,
    #                                         ROIs_analyze=ROIs_analyze,
    #                                         group_avg=True)

    cols = ['dv_ant', 'dv_pos', 'dpva', 'vpda']
    # print(list(df.columns))
    # quit()

    df['vendor'] = df['dv_ant'] + df['dv_pos'] - df['dpva'] - df['vpda']
    # df['vendor'] = df['dpva'] + df['vpda']
    # df['vendor'] = np.abs(df['vendor'])

    l = []
    for sn, df_sn in df.groupby('sn'):
        if sn not in sn2scores:
            continue
        scores = sn2scores[sn]
        # print(scores)
        # quit()
        r, p = stats.spearmanr(df_sn['vendor'], scores, nan_policy='omit')
        print(f'{r=:.3f}')
        l.append(r)
    M = np.mean(l)
    SD = np.std(l)
    N = len(l)
    SE = SD / np.sqrt(N)
    t = M / SE
    p = stats.t.sf(np.abs(t), N-1) * 2
    print(f'{M=:.3f}, {t=:.3f}, {p=:.3f}')

if __name__ == '__main__':
    ROIs_analyze = ['SFG', 'MFG', 'IFG', 'OrG', 'PrG', 'PCL', 'pSTS', 'SPL',
                    'IPL', 'Pcun', 'PoG', 'INS', 'CG', 'Amyg', 'Hipp', 'Str',
                    'Tha', 'ACC', 'PCC']

    ROIs_analyze = ['STG', 'MTG', 'Pcun', 'PCL', 'Str', 'pSTS', 'PCC', 'PrG',
                    'sOcG', 'PhG', 'ATL', 'IPL', 'OrG', 'EVC', 'IFG', 'PoG',
                    'Amyg', 'Hipp', 'Tha', 'INS', 'SPL', 'FuG', 'SFG', 'ACC',
                    'MFG', 'ITG', 'LOC']

    # ROIs_analyze = ['ITG', 'MTG', 'FuG', 'PhG', 'ATL', 'Hipp']
    #
    # ROIs_analyze = ['STG', 'MTG', 'Pcun', 'PCL', 'pSTS', 'PCC', 'PrG',
    #                 'sOcG', 'PhG', 'ATL', 'IPL', 'OrG', 'EVC', 'IFG', 'PoG',
    #                 'INS', 'SPL', 'FuG', 'SFG', 'ACC', 'MFG', 'ITG', 'LOC']
    #
    # ROIs_analyze = ['LOC', 'EVC', 'sOcG']

    # ROIs_analyze = ['LOC', 'EVC', 'sOcG', 'ITG', 'MTG', 'FuG', 'PhG', 'ATL', 'Hipp']


    do_episodic_x_vd(ROIs_analyze=ROIs_analyze, mem='con')
    # do_episodic_x_vd(ROIs_analyze=ROIs_analyze, mem='vis')

