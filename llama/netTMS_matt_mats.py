from collections import defaultdict

from scipy import io, stats
import pandas as pd
import numpy as np
from functools import cache
from scipy.spatial.distance import pdist, squareform
import os
from marinate.pkld import pkld
from tqdm import tqdm
import shutil


@pkld
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


def load_stimID_cvs_l(layers, activation_model='Llama-3.2-3b'):
    d_all = defaultdict(list)
    for layer in layers:
        d_layer = load_stimID_cvs(layer, activation_model=activation_model)
        for stimID, vec in d_layer.items():
            d_all[stimID].append(vec)
    for stimID, vecs in d_all.items():
        d_all[stimID] = np.concatenate(vecs)
    return d_all

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

if __name__ == '__main__':
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
        print(f'ROI{ROI:03d} | t[{N-1}]={t:>5.2f} {p=:.3f} {stars}')
        ts_all.append(t)
    M_all = np.mean(ts_all)
    print(f'{M_all=}')
    M_all_abs = np.mean(np.abs(ts_all))
    print(f'{M_all_abs=}')
