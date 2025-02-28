import os
import shutil
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import io, stats
from tqdm import tqdm

from marinate.pkld import pkld


@pkld(store='both')
def load_stimID_cvs(layer, activation_model='Llama-3.2-3b'):
    fp_in = (fr'C:\PycharmProjects\SchemeRep\llama\vec_csvs_DinoLab\Llama-3.2-3b'
             fr'\input\vecs_{activation_model}_input_{layer}.csv')
    df = pd.read_csv(fp_in)
    d = {}
    for idx, row in df.iterrows():
        stimID = row['ID']
        stim_vec = []
        for dim in range(3072):
            val = row[f'dim_{dim}']
            stim_vec.append(val)
        d[stimID] = np.array(stim_vec)
    return d


@pkld(store='memory')
def load_stimID_cvs_l(layers, activation_model='Llama-3.2-3b'):
    d_all = defaultdict(list)
    for layer in layers:
        d_layer = load_stimID_cvs(layer, activation_model=activation_model)
        for stimID, vec in d_layer.items():
            d_all[stimID].append(vec)
    for stimID, vecs in d_all.items():
        d_all[stimID] = np.concatenate(vecs)
    return d_all


@pkld(store='memory')
def get_llama_RSM_from_stimIDs(stimIDs, layer=7, norm=True):
    if isinstance(layer, int):
        d = load_stimID_cvs(layer)
    else:
        d = load_stimID_cvs_l(layer)
    vecs = np.array([d[stimID] for stimID in stimIDs]).T
    if norm:
        vecs = stats.zscore(vecs, axis=1)
    RSM = stats.spearmanr(vecs).correlation
    return RSM


def copy_in():
    sns_in = [5001, 5002, 5004, 5005, 5007, 5010, 5011, 5012, 5014, 5015,
              5016, 5017, 5020, 5021, 5022, 5025, 5026, 5028, 5029, 5030,
              5031, 5032, 5033]
    for sn in tqdm(sns_in, desc='Copying in sns'):
        print(f'Onto: {sn}')
        dir_in = fr'M:\Simon\NetTMS.01\Analysis\RSA\data\neuralRDMs\HOA\{sn}'
        assert os.path.exists(dir_in)
        dir_out = fr'C:\PycharmProjects\SchemeRep\fMRI_in\NetTMS_RDMs\HOA\{sn}'
        if os.path.exists(dir_out):
            print(f'Already exists: {sn} dir')
            continue
        shutil.copytree(dir_in, dir_out)
    quit()


@pkld()
def get_ROI_rs(roi, layer=7, norm=True):
    ROI_rs = []
    for sn in sns_in:
        fp_roi_sn = fr'C:\PycharmProjects\SchemeRep\fMRI_in\NetTMS_RDMs\HOA\{sn}\{sn}_ROI{roi:03d}_Day{DAY}.mat'
        if not os.path.exists(fp_roi_sn):
            continue
        mat = io.loadmat(fp_roi_sn)
        NSM = mat['R']
        stimIDs = mat['stimID'][0]
        RSM = get_llama_RSM_from_stimIDs(stimIDs, layer=layer,
                                         norm=norm)
        trils = np.tril_indices_from(RSM, k=-1)
        NSM_flat = NSM[trils]
        RSM_flat = RSM[trils]
        r = stats.spearmanr(NSM_flat, RSM_flat).correlation
        if np.isnan(r):
            continue
        ROI_rs.append(r)
        # print(f'bad, ROI{ROI:03d} | {sn} | {r=}')
    return ROI_rs


def get_IRAFs(fMRI_RDM, stim_RDM, second_order='spear'):
    '''
    matmul all took 0.001 seconds
    matmul semi (one loop, inner matmul) took 0.008 seconds
    scipy pearsonr took 0.013 seconds
    scipy spearmanr took 0.055 seconds
    '''
    fMRI_RDM_ = fMRI_RDM.copy()
    stim_RDM_ = stim_RDM.copy()

    fMRI_RDM_[np.diag_indices_from(fMRI_RDM)] = np.nan
    stim_RDM_[np.diag_indices_from(stim_RDM)] = np.nan
    stim_RDM_ = stim_RDM.copy()
    if second_order == 'corr':
        fMRI_RDM_std = stats.zscore(fMRI_RDM_, axis=0, nan_policy='omit')
        stim_RDM_std = stats.zscore(stim_RDM_, axis=0, nan_policy='omit')
        # fMRI_RDM_std = stdize(fMRI_RDM_, axis=0, nans=True)
        # stim_RDM_std = stdize(stim_RDM, axis=0, nans=True)
        IRAFs = np.nanmean(fMRI_RDM_std * stim_RDM_std, axis=0)
        IRAFs = np.arctanh(IRAFs)
    elif second_order == 'spear':
        fMRI_RDM_[np.isnan(stim_RDM_)] = np.nan
        stim_RDM_[np.isnan(fMRI_RDM_)] = np.nan
        fMRI_RDM_r = stats.rankdata(fMRI_RDM_, axis=0, nan_policy='omit')
        stim_RDM_r = stats.rankdata(stim_RDM_, axis=0, nan_policy='omit')
        # fMRI_RDM_r = stdize(fMRI_RDM_r, axis=0, nans=True)
        # stim_RDM_r = stdize(stim_RDM_r, axis=0, nans=True)
        fMRI_RDM_r = stats.zscore(fMRI_RDM_r, axis=0, nan_policy='omit')
        stim_RDM_r = stats.zscore(stim_RDM_r, axis=0, nan_policy='omit')
        IRAFs = np.nanmean(fMRI_RDM_r * stim_RDM_r, axis=0)
        IRAFs = np.arctanh(IRAFs)
    else:
        raise NotImplementedError(f'get_IRAFs {second_order=}')
    return IRAFs


@pkld
def get_IRAFs_sn_roi_day(sn, roi, day, layer=tuple(range(4, 12))):
    if isinstance(layer, tuple):
        layer = list(layer)
    fp_roi_sn = fr'C:\PycharmProjects\SchemeRep\fMRI_in\NetTMS_RDMs\HOA\{sn}\{sn}_ROI{roi:03d}_Day{day}.mat'
    mat = io.loadmat(fp_roi_sn)
    NSM = mat['R']
    stimIDs = mat['stimID'][0]
    RSM = get_llama_RSM_from_stimIDs(stimIDs, layer=layer,
                                     norm=True)

    IRAFs = get_IRAFs(NSM, RSM, second_order='spear')
    return IRAFs


def make_matt_csv(layer_low=4, layer_high=12):
    sns_in = [5001, 5002, 5004, 5005, 5007, 5010, 5011, 5012, 5014, 5015,
              5016, 5017, 5020, 5021, 5022, 5025, 5026, 5028, 5029, 5030,
              5031, 5032, 5033]
    dfs_all = []
    for sn in tqdm(sns_in, desc='Cooking subjects'):
        for day in range(1, 5):
            print(f'Onto: {sn}, {day}')
            roi2IRAFs = {}
            for roi in range(1, 471):
                try:
                    IRAFs = get_IRAFs_sn_roi_day(sn, roi, day,
                                                 layer=list(range(layer_low, layer_high)))
                except FileNotFoundError:
                    print(f'Missing file: {sn}/{day}')
                    break
                roi2IRAFs[f'ROI{roi:03d}'] = IRAFs
            else:
                fp_roi_sn = fr'C:\PycharmProjects\SchemeRep\fMRI_in\NetTMS_RDMs\HOA\{sn}\{sn}_ROI{roi:03d}_Day{day}.mat'
                mat = io.loadmat(fp_roi_sn)
                stimIDs = mat['stimID'][0]
                df_day = pd.DataFrame(roi2IRAFs)
                df_day['sn'] = sn
                df_day['day'] = day
                df_day['stimID'] = stimIDs
                dfs_all.append(df_day)
    df_all = pd.concat(dfs_all)
    fp_out = fr'C:\PycharmProjects\SchemeRep\matthew_NetTMS_IRAFs_layers{layer_low}-{layer_high}.csv'
    df_all.to_csv(fp_out, index=False)
    quit()


if __name__ == '__main__':
    make_matt_csv()
    # copy_in()
    sns_in = [5001, 5002, 5004, 5005, 5007, 5010, 5011, 5012, 5014, 5015,
              5016, 5017, 5020, 5021, 5022, 5025, 5026, 5028, 5029, 5030,
              5031, 5032, 5033]

    DAY = 1

    # 5029 is missing day 1

    ts_all = []
    for ROI in range(1, 471):
        ROI_rs = get_ROI_rs(ROI, layer=list(range(6, 12)))
        if len(ROI_rs) < 2:
            print(f'All bad: {ROI:03d}, {len(ROI_rs)=}')
            continue
        t, p = stats.ttest_1samp(ROI_rs, 0)
        N = len(ROI_rs)
        # {np.mean(ROI_rs):.3f} |
        if t > 4:
            stars = '***'
        elif t > 3:
            stars = '**'
        elif t > 2:
            stars = '*'
        else:
            stars = ''
        print(f'ROI{ROI:03d} | t[{N - 1}]={t:>5.2f} {p=:.3f} {stars}')
        ts_all.append(t)
    M_all = np.mean(ts_all)
    print(f'{M_all=}')
    M_all_abs = np.mean(np.abs(ts_all))
    print(f'{M_all_abs=}')
