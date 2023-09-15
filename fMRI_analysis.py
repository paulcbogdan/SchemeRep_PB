import random

from pickle_wrap import pickle_wrap
from collections import defaultdict

import numpy as np

from DNN_vectors import get_DNN_vecs, get_img_fns
from organize_bhv import get_trial_info
from nilearn import image, datasets
from glob import glob

from wordvec_get_vectors import get_semantic_vectors
import matplotlib.pyplot as plt
import utils
import scipy.stats as stats

from scipy import io
import pandas as pd
from tqdm import tqdm
from pathlib import Path


def get_stim_RDM(df_sn, d_vecs, obj_only=False, scene_only=False,
                 dif=True):
    # return get_stim_RDM_lifu(df_sn)
    vec_size = len(d_vecs[df_sn['obj'].iloc[0]])
    # all_vecs = np.empty((len(df_sn['obj']), vec_size))
    vecs_obj = np.empty((len(df_sn['obj']), vec_size))
    vecs_scene = np.empty((len(df_sn['obj']), vec_size))
    for i, (obj, scene, obj_rename, scene_rename) in enumerate(zip(df_sn['obj'],
            df_sn['scene'], df_sn['obj_rename'], df_sn['scene_rename'])):
        vec_obj = d_vecs[obj]
        vec_scene = d_vecs[scene]
        vecs_obj[i, :] = vec_obj
        vecs_scene[i, :] = vec_scene
        # if obj_only:
        #     # obj = obj.replace(' ', '_')
        #     vec_obj = d_vecs[obj]
        #     all_vecs[i, :] = vec_obj
        #     vecs_obj[i, :] = vec_obj
        # elif scene_only:
        #     # scene = scene.replace(' ', '_')
        #     vec_scene = d_vecs[scene]
        #     all_vecs[i, :] = vec_scene
        #     vecs_scene[i, :] = vec_scene
        # else:
        #     # obj = obj.replace(' ', '_')
        #     # scene = scene.replace(' ', '_')
        #     vec_obj = abs(d_vecs[obj] - d_vecs[scene])
        #     all_vecs[i, :] = vec_obj

    if obj_only:
        all_vecs = vecs_obj
    elif scene_only:
        all_vecs = vecs_scene
    else:
        # obj_mean = np.mean(vecs_obj, axis=0)
        # scene_mean = np.mean(vecs_scene, axis=0)
        # all_vecs = np.abs(vecs_obj - obj_mean - vecs_scene + scene_mean)
        all_vecs = abs(vecs_obj - vecs_scene)# ** 2


    RDM_stim = np.corrcoef(all_vecs)
    pd.DataFrame(RDM_stim).to_csv('RDM_stim_mine.csv')
    # get_stim_RDM_lifu(df_sn)
    # quit()
    return RDM_stim

def get_stim_RDM_lifu(df_sn):
    print('Loading existing...')
    fp_in = r'C:\PycharmProjects_C\SchemeRep\RSAmodels\example_deepNeuralNetworkScripts_from_Lifu\RSAmodel\modelRDMs' \
            r'\RSM_VGG16_PCA.mat'
    mat = io.loadmat(fp_in)
    RDM_stim = mat['R']
    RDM_new = np.zeros((len(df_sn), len(df_sn)))

    tblStim = pd.read_csv(r"SchemRep_tasks\PTBtasks\fullStimList.csv")
    tblStim.head()
    filelist = tblStim['ObjectFile'].to_list()
    name2fps = get_img_fns(get_dict=True)

    for obj0 in tqdm(df_sn['obj'], desc='prepping Lifu RDM'):
        obj0 = name2fps[obj0].replace(r'SchemRep_tasks\PTBtasks\updatedObjectsResampled', '')[1:]
        for obj1 in df_sn['obj']:
            obj1 = name2fps[obj1].replace(r'SchemRep_tasks\PTBtasks\updatedObjectsResampled', '')[1:]
            idx0 = filelist.index(obj0)
            idx1 = filelist.index(obj1)
            RDM_new[idx0, idx1] = RDM_stim[idx0, idx1]
            RDM_new[idx1, idx0] = RDM_stim[idx1, idx0]
    pd.DataFrame(RDM_new).to_csv('RDM_stim_lifu.csv')
    # quit()

    return RDM_new




def get_atlas_resampled(combine_bilateral=True):
    fp_ref = r'Day2EncSingleTrialModellingLSS_sorted/102/all_ENCruns_sorted/objects/Day2_Run1_Trial4_UnifiedID53_StimID215_Subset2_pairID15_Con3_Resp4_IsObject1.nii'
    img = image.load_img(fp_ref)
    atlas = utils.get_BN_atlas()
    atlas['maps'] = image.resample_to_img(atlas['maps'], img,
                                          interpolation='nearest')
    return atlas

def stdize(v, axis=None):
    return (v - np.mean(v, axis=axis)) / np.std(v, axis=axis)

def RDM_x_RDM(fMRI_RDM, stim_RDM):
    assert fMRI_RDM.shape == stim_RDM.shape, 'RDMs must be the same shape: ' \
       f'fMRI_RDM.shape = {fMRI_RDM.shape}, stim_RDM.shape = {stim_RDM.shape}'
    tril_idx = np.tril_indices_from(fMRI_RDM, k=-1)
    fMRI_vec = 1 - fMRI_RDM[tril_idx]
    # print(fMRI_vec.shape)
    # quit()
    stim_vec = 1 - stim_RDM[tril_idx]
    # fMRI_M = np.mean(fMRI_vec)
    # return fMRI_M
    r, _ = stats.spearmanr(fMRI_vec, stim_vec)
    # r = np.corrcoef(fMRI_vec, stim_vec)[0, 1]
    z = np.arctanh(r)
    return z

def get_IRAFs(fMRI_RDM, stim_RDM, df_sn, flipper=0):
    IRAFs = []
    for i in range(fMRI_RDM.shape[0]):
        fMRI_vec_std = np.delete(fMRI_RDM[i, :], i)
        stim_vec_std = np.delete(stim_RDM[i, :], i)
        fMRI_vec_std = stdize(fMRI_vec_std) # exclude correlation w itself
        stim_vec_std = stdize(stim_vec_std)
        r = (fMRI_vec_std @ stim_vec_std) / len(fMRI_vec_std)
        IRAFs.append(r)
    IRAFs = [IRAF for (IRAF, _) in sorted(zip(IRAFs, df_sn['obj']),
                                          key=lambda x: x[1])]
    return IRAFs


def explore_hist(stim_RDM):
    tril_idx = np.tril_indices_from(stim_RDM, k=-1)
    stim_vec = stim_RDM[tril_idx]
    vmin = np.quantile(stim_vec, .005)
    vmax = np.quantile(stim_vec, .995)
    fig, axs = plt.subplots(2, 1, figsize=(10, 5))
    axs[0].imshow(stim_RDM, vmin=vmin, vmax=vmax, cmap='turbo')
    axs[1].hist(stim_vec, bins=100, range=(vmin, vmax))
    plt.show()


def get_all_sns():
    age2sn = defaultdict(list)
    bad_sns = {'126', '131',
               '201', '224', '231', '232', '233', '234', '235'}
    for age in range(1, 4):
        bhv_root = fr'behavFiles/ENC/S{age}*_run1.mat'
        fns = glob(bhv_root)
        for fn in fns:
            sn = fn.replace('behavFiles/ENC\\S', '').replace('_run1.mat', '')
            # sn = fn.replace(r'behavFiles/ENC/S', '').replace('_run1.mat', '')
            if sn in bad_sns:
                continue
            age2sn[age].append(sn)
    age2sn['healthy'] = age2sn[1] + age2sn[2]
    return age2sn

def print_ROI_to_results(ROI_to_results):
    print()
    for ROI, results in ROI_to_results.items():
        if len(results) < 2:
            continue
        M = np.mean(results)
        SD = np.std(results)
        SE = SD / np.sqrt(len(results))
        t = M / SE
        print(f'{ROI}: {M=:.3f} [{SE=:.3f}], {t=:.3f}, N = {len(results)}')

def get_ROI_info():
    ROIs_HC = ['215 Hipp_L_2_1', '216 Hipp_R_2_1', '217 Hipp_L_2_2', '218 Hipp_R_2_2']
    ROIs_MOC = ['189 Cun_L_5_1', '190 Cun_R_5_1', '191 Cun_L_5_2', '192 Cun_R_5_2',
            '193 Cun_L_5_3', '194 Cun_R_5_3', '195 Cun_L_5_4', '196 Cun_R_5_4',
            '197 Cun_L_5_5', '198 Cun_R_5_5']
    ROIs_LOC = ['199 OcG_L_4_1', '200 OcG_R_4_1',
            '201 OcG_L_4_2', '202 OcG_R_4_2', '203 OcG_L_4_3', '204 OcG_R_4_3',
            '205 OcG_L_4_4', '206 OcG_R_4_4', '207 sOcG_L_2_1', '208 sOcG_R_2_1',
            '209 sOcG_L_2_2', '210 sOcG_R_2_2']
    ROIs_all = ['1 SFG_L_7_1', '2 SFG_R_7_1', '3 SFG_L_7_2', '4 SFG_R_7_2',
            '5 SFG_L_7_3', '6 SFG_R_7_3', '7 SFG_L_7_4', '8 SFG_R_7_4',
            '9 SFG_L_7_5', '10 SFG_R_7_5', '11 SFG_L_7_6', '12 SFG_R_7_6',
            '13 SFG_L_7_7', '14 SFG_R_7_7', '15 MFG_L_7_1', '16 MFG_R_7_1',
            '17 MFG_L_7_2', '18 MFG_R_7_2', '19 MFG_L_7_3', '20 MFG_R_7_3',
            '21 MFG_L_7_4', '22 MFG_R_7_4', '23 MFG_L_7_5', '24 MFG_R_7_5',
            '25 MFG_L_7_6', '26 MFG_R_7_6', '27 MFG_L_7_7', '28 MFG_R_7_7',
            '29 IFG_L_6_1', '30 IFG_R_6_1', '31 IFG_L_6_2', '32 IFG_R_6_2',
            '33 IFG_L_6_3', '34 IFG_R_6_3', '35 IFG_L_6_4', '36 IFG_R_6_4',
            '37 IFG_L_6_5', '38 IFG_R_6_5', '39 IFG_L_6_6', '40 IFG_R_6_6',
            '41 OrG_L_6_1', '42 OrG_R_6_1', '43 OrG_L_6_2', '44 OrG_R_6_2',
            '45 OrG_L_6_3', '46 OrG_R_6_3', '47 OrG_L_6_4', '48 OrG_R_6_4',
            '49 OrG_L_6_5', '50 OrG_R_6_5', '51 OrG_L_6_6', '52 OrG_R_6_6',
            '53 PrG_L_6_1', '54 PrG_R_6_1', '55 PrG_L_6_2', '56 PrG_R_6_2',
            '57 PrG_L_6_3', '58 PrG_R_6_3', '59 PrG_L_6_4', '60 PrG_R_6_4',
            '61 PrG_L_6_5', '62 PrG_R_6_5', '63 PrG_L_6_6', '64 PrG_R_6_6',
            '65 PCL_L_2_1', '66 PCL_R_2_1', '67 PCL_L_2_2', '68 PCL_R_2_2',
            '69 STG_L_6_1', '70 STG_R_6_1', '71 STG_L_6_2', '72 STG_R_6_2',
            '73 STG_L_6_3', '74 STG_R_6_3', '75 STG_L_6_4', '76 STG_R_6_4',
            '77 STG_L_6_5', '78 STG_R_6_5', '79 STG_L_6_6', '80 STG_R_6_6',
            '81 MTG_L_4_1', '82 MTG_R_4_1', '83 MTG_L_4_2', '84 MTG_R_4_2',
            '85 MTG_L_4_3', '86 MTG_R_4_3', '87 MTG_L_4_4', '88 MTG_R_4_4',
            '89 ITG_L_7_1', '90 ITG_R_7_1', '91 ITG_L_7_2', '92 ITG_R_7_2',
            '93 ITG_L_7_3', '94 ITG_R_7_3', '95 ITG_L_7_4', '96 ITG_R_7_4',
            '97 ITG_L_7_5', '98 ITG_R_7_5', '99 ITG_L_7_6', '100 ITG_R_7_6',
            '101 ITG_L_7_7', '102 ITG_R_7_7', '103 FuG_L_3_1', '104 FuG_R_3_1',
            '105 FuG_L_3_2', '106 FuG_R_3_2', '107 FuG_L_3_3', '108 FuG_R_3_3',
            '109 PhG_L_6_1', '110 PhG_R_6_1', '111 PhG_L_6_2', '112 PhG_R_6_2',
            '113 PhG_L_6_3', '114 PhG_R_6_3', '115 PhG_L_6_4', '116 PhG_R_6_4',
            '117 PhG_L_6_5', '118 PhG_R_6_5', '119 PhG_L_6_6', '120 PhG_R_6_6',
            '121 pSTS_L_2_1', '122 pSTS_R_2_1', '123 pSTS_L_2_2', '124 pSTS_R_2_2',
            '125 SPL_L_5_1', '126 SPL_R_5_1', '127 SPL_L_5_2', '128 SPL_R_5_2',
            '129 SPL_L_5_3', '130 SPL_R_5_3', '131 SPL_L_5_4', '132 SPL_R_5_4',
            '133 SPL_L_5_5', '134 SPL_R_5_5', '135 IPL_L_6_1', '136 IPL_R_6_1',
            '137 IPL_L_6_2', '138 IPL_R_6_2', '139 IPL_L_6_3', '140 IPL_R_6_3',
            '141 IPL_L_6_4', '142 IPL_R_6_4', '143 IPL_L_6_5', '144 IPL_R_6_5',
            '145 IPL_L_6_6', '146 IPL_R_6_6', '147 Pcun_L_4_1', '148 Pcun_R_4_1',
            '149 Pcun_L_4_2', '150 Pcun_R_4_2', '151 Pcun_L_4_3', '152 Pcun_R_4_3',
            '153 Pcun_L_4_4', '154 Pcun_R_4_4', '155 PoG_L_4_1', '156 PoG_R_4_1',
            '157 PoG_L_4_2', '158 PoG_R_4_2', '159 PoG_L_4_3', '160 PoG_R_4_3',
            '161 PoG_L_4_4', '162 PoG_R_4_4', '163 INS_L_6_1', '164 INS_R_6_1',
            '165 INS_L_6_2', '166 INS_R_6_2', '167 INS_L_6_3', '168 INS_R_6_3',
            '169 INS_L_6_4', '170 INS_R_6_4', '171 INS_L_6_5', '172 INS_R_6_5',
            '173 INS_L_6_6', '174 INS_R_6_6', '175 CG_L_7_1', '176 CG_R_7_1',
            '177 CG_L_7_2', '178 CG_R_7_2', '179 CG_L_7_3', '180 CG_R_7_3',
            '181 CG_L_7_4', '182 CG_R_7_4', '183 CG_L_7_5', '184 CG_R_7_5',
            '185 CG_L_7_6', '186 CG_R_7_6', '187 CG_L_7_7', '188 CG_R_7_7',
            '189 Cun_L_5_1', '190 Cun_R_5_1', '191 Cun_L_5_2', '192 Cun_R_5_2',
            '193 Cun_L_5_3', '194 Cun_R_5_3', '195 Cun_L_5_4', '196 Cun_R_5_4',
            '197 Cun_L_5_5', '198 Cun_R_5_5', '199 OcG_L_4_1', '200 OcG_R_4_1',
            '201 OcG_L_4_2', '202 OcG_R_4_2', '203 OcG_L_4_3', '204 OcG_R_4_3',
            '205 OcG_L_4_4', '206 OcG_R_4_4', '207 sOcG_L_2_1', '208 sOcG_R_2_1',
            '209 sOcG_L_2_2', '210 sOcG_R_2_2', '211 Amyg_L_2_1', '212 Amyg_R_2_1',
            '213 Amyg_L_2_2', '214 Amyg_R_2_2', '215 Hipp_L_2_1', '216 Hipp_R_2_1',
            '217 Hipp_L_2_2', '218 Hipp_R_2_2', '219 Str_L_6_1', '220 Str_R_6_1',
            '221 Str_L_6_2', '222 Str_R_6_2', '223 Str_L_6_3', '224 Str_R_6_3',
            '225 Str_L_6_4', '226 Str_R_6_4', '227 Str_L_6_5', '228 Str_R_6_5',
            '229 Str_L_6_6', '230 Str_R_6_6', '231 Tha_L_8_1', '232 Tha_R_8_1',
            '233 Tha_L_8_2', '234 Tha_R_8_2', '235 Tha_L_8_3', '236 Tha_R_8_3',
            '237 Tha_L_8_4', '238 Tha_R_8_4', '239 Tha_L_8_5', '240 Tha_R_8_5',
            '241 Tha_L_8_6', '242 Tha_R_8_6', '243 Tha_L_8_7', '244 Tha_R_8_7',
            '245 Tha_L_8_8', '246 Tha_R_8_8']
    ROIs = ROIs_all

    ROI_nums = [int(roi.split()[0]) for roi in ROIs]
    n_ROIs = len(ROIs)

    region_nums = defaultdict(list)
    for ROI in ROIs:
        ROI_num, ROI_str = ROI.split(' ')
        region = ROI_str.split('_')[0]
        region_nums[region].append(int(ROI_num)-1)
    ticks = []
    tick_labels = []
    tick_lows = []
    for region, l in region_nums.items():
        ticks.append(np.mean(l))
        tick_labels.append(region)
        tick_lows.append(l[0])

    return ROIs, ROI_nums, ticks, tick_labels, tick_lows, n_ROIs

def regress_out(x, y):
    x = np.array(x)
    y = np.array(y)
    nans = np.isnan(x) | np.isnan(y)
    if len(x[~nans]) < 3:
        return y
    b, m, r, p, er = stats.linregress(x[~nans], y[~nans])
    # print(f'{x=}')
    # print(f'{r=} | {b=}')
    return y - x*b

def get_IRAF_connectivity_matrix(ROI_to_IRAF, n_ROIs, ROIs):
    # TODO: optimize to run correlations on everyone simultaneously
    first_key = next(iter(ROI_to_IRAF.keys()))
    subj_timeseries_shape = (n_ROIs, ROI_to_IRAF[first_key].shape[1])
    subj2timeseries = defaultdict(lambda: np.full(subj_timeseries_shape,
                                                  np.nan))
    print(f'{len(ROI_to_IRAF)}')
    for i, ROI in enumerate(ROIs):
        ar = ROI_to_IRAF[ROI]
        for subj_j in range(ar.shape[0]):
            subj2timeseries[subj_j][i, :] = ar[subj_j, :]
    all_matricies = []
    for sn, ar in subj2timeseries.items():
        # avg_over_time = np.nanmean(ar, axis=0)
        # std_over_time = np.nanstd(ar, axis=0)
        # ar = (ar - avg_over_time) / std_over_time
        # print(ar.shape)
        # ar = np.append(ar, ar.mean(axis=1, keepdims=True), axis=1)
        # print(ar.shape)
        # print(ar.mean(axis=1, keepdims=True))
        # max_val = np.max(np.abs(ar))
        # print(f'{max_val=}')
        # plt.imshow(ar)
        # plt.title(f'Plot IRAF by ROI and stimulus: {sn}')
        # plt.ylabel('ROI')
        # plt.xlabel('Stimulus')
        # plt.colorbar()
        # plt.show()
        mat = np.corrcoef(ar)
        mat[np.diag_indices_from(mat)] = 0
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if i % 2 == j % 2:
                    if (i < mat.shape[0] - 1):
                        mat[i, j] = mat[i+1, j]
        # plt.imshow(mat)
        # plt.title('correlation matrix')
        # plt.colorbar()
        # plt.show()
        mat[mat > .99] = .99
        mat[mat < -.99] = -.99
        mat = np.arctanh(mat)
        all_matricies.append(mat)
    # quit()
    return all_matricies

def calculate_triple_connectivity(ROI_to_RDM_fMRI, RDM_stim, ROIs):
    trils = np.tril_indices_from(ROI_to_RDM_fMRI[ROIs[0]], k=-1)
    for ROI, rdm in ROI_to_RDM_fMRI.items():
        ROI_to_RDM_fMRI[ROI] = (rdm - np.nanmean(rdm[trils])) / np.nanstd(rdm[trils])
    RDM_stim = (RDM_stim - np.nanmean(RDM_stim[trils])) / np.nanstd(RDM_stim[trils])

    triple_prod_mat = np.zeros((len(ROIs), len(ROIs)))
    for j, (ROI0, RDM0) in enumerate(ROI_to_RDM_fMRI.items()):
        for k, (ROI1, RDM1) in enumerate(ROI_to_RDM_fMRI.items()):
            if j >= k: continue
            prod = np.multiply(np.multiply(RDM0, RDM1), RDM_stim)
            prod = prod[trils]
            triple_prod_mat[ROIs.index(ROI1), ROIs.index(ROI0)] = \
                triple_prod_mat[ROIs.index(ROI0), ROIs.index(ROI1)] = \
                np.nanmean(prod)
    return triple_prod_mat

def get_vector_product():
    pass

def mass_RDM_x_RDM(age=1, cin=None, semantic=False, early=False):
    if semantic:
        d_vecs = get_semantic_vectors()
    else:
        d_vecs = get_DNN_vecs(early=early, PCA=True)

    ROIs, ROI_nums, ticks, tick_labels, tick_lows, n_ROIs = get_ROI_info()

    atlas = get_atlas_resampled()
    age2sn = get_all_sns()

    ROI_to_dif_ = defaultdict(list)
    ROI_to_dif = defaultdict(list)
    ROI_to_obj = defaultdict(list)
    ROI_to_scn = defaultdict(list)
    n_trials = 114 if cin is None else 38
    # n_trials *= 2
    ROI_to_IRAF_dif_ = defaultdict(lambda: np.full((0, n_trials), np.nan))
    ROI_to_IRAF_dif = defaultdict(lambda: np.full((0, n_trials), np.nan))
    ROI_to_IRAF_obj = defaultdict(lambda: np.full((0, n_trials), np.nan))
    ROI_to_IRAF_scn = defaultdict(lambda: np.full((0, n_trials), np.nan))
    ROI_to_activity = defaultdict(list)

    triple_z = {'obj': [], 'scn': [], 'dif': []}
    rxr = {'obj': [], 'scn': [], 'dif': []}

    Zs_all_ = []
    # test = get_trial_info('138')
    # for i, name in enumerate(sorted(test['obj'])):
    #     print(f'{i}: {name}')
    vecs = list(d_vecs.values())
    random.shuffle(vecs)
    # for i, (name, vec) in enumerate(d_vecs.items()):
    #     d_vecs[name] = vecs[i]
    for i, sn in tqdm(enumerate(age2sn[age]),
                      desc=f'Looping subjects: age2sn[{age}]'):
        df_sn = get_trial_info(sn)
        if cin is not None:
            df_sn = df_sn[df_sn['CIN'] == cin]
        RDM_stim_obj = get_stim_RDM(df_sn, d_vecs, obj_only=True)
        RDM_stim_scn = get_stim_RDM(df_sn, d_vecs, obj_only=False,
                                      scene_only=True)
        RDM_stim_dif = get_stim_RDM(df_sn, d_vecs, obj_only=False, dif=True)
        img = image.load_img(df_sn['fp_fMRI']).get_fdata()
        ROI_to_RDM_fMRI = {}
        rxr_mats = {'obj': [], 'scn': [], 'dif': []}
        ROI2vecs = {}
        for j, (ROI, ROI_num) in enumerate(zip(ROIs, ROI_nums)):
            region_vecs = img[atlas['maps'].get_fdata() == ROI_num]
            voxels_w_nan = np.isnan(region_vecs).any(axis=1)
            n_nans_ROI = np.sum(voxels_w_nan)
            if n_nans_ROI / len(voxels_w_nan) > 0.25: # more than 10%
                continue
            region_vecs = region_vecs[~voxels_w_nan, :]
            ROI2vecs[ROI] = region_vecs

        for j, (ROI, ROI_num) in enumerate(zip(ROIs, ROI_nums)):
            if ROI not in ROI2vecs:
                for key in ['obj', 'scn', 'dif']:
                    rxr_mats[key].append([np.nan] * len(ROIs))
                continue
            region_vecs = ROI2vecs[ROI]

            mean_activity = np.nanmean(region_vecs, axis=0)
            mean_activity = [a for (a, _) in sorted(zip(mean_activity,
                                                        df_sn['obj']),
                                                  key=lambda x: x[1])]

            RDM_fMRI = np.corrcoef(region_vecs.T)
            ROI_to_RDM_fMRI[ROI] = RDM_fMRI
            z_RDM_x_RDM_obj = RDM_x_RDM(RDM_fMRI, RDM_stim_obj)
            z_RDM_x_RDM_scene = RDM_x_RDM(RDM_fMRI, RDM_stim_scn)
            z_RDM_x_RDM_dif = RDM_x_RDM(RDM_fMRI, RDM_stim_dif)

            nans_RDM_obj = np.sum(np.isnan(RDM_stim_obj))
            nans_RDM_scn = np.sum(np.isnan(RDM_stim_scn))
            if nans_RDM_obj > 0 or nans_RDM_scn > 0:
                print(f'{nans_RDM_obj=}, {nans_RDM_scn=}')

            IRAFs_obj = get_IRAFs(RDM_fMRI, RDM_stim_obj, df_sn)
            IRAFs_scene = get_IRAFs(RDM_fMRI, RDM_stim_scn, df_sn)
            IRAFs_dif = get_IRAFs(RDM_fMRI, RDM_stim_dif, df_sn)
            if np.isnan(z_RDM_x_RDM_dif):
                print('HOW NaN?')
                continue

            ROI_to_IRAF_dif[ROI] = np.append(ROI_to_IRAF_dif[ROI],
                                             [IRAFs_dif], axis=0)
            ROI_to_IRAF_scn[ROI] = np.append(ROI_to_IRAF_scn[ROI],
                                               [IRAFs_scene], axis=0)
            ROI_to_IRAF_obj[ROI] = np.append(ROI_to_IRAF_obj[ROI],
                                             [IRAFs_obj], axis=0)

            ROI_to_dif[ROI].append(z_RDM_x_RDM_dif)
            ROI_to_obj[ROI].append(z_RDM_x_RDM_obj)
            ROI_to_scn[ROI].append(z_RDM_x_RDM_scene)
            ROI_to_activity[ROI].append(mean_activity)

            # print('Onto rxr RDM...')
            # region_vecs_ = np.expand_dims(region_vecs, 2)
            # rxr_vecs = {'obj': [], 'scn': [], 'dif': []}
            # for k, (ROI1, ROI_num1) in tqdm(enumerate(zip(ROIs, ROI_nums)), desc='test'):
            #     if ROI1 not in ROI2vecs:
            #         for key in ['obj', 'scn', 'dif']:
            #             rxr_vecs[key].append(np.nan)
            #         continue
            #     region_vecs1 = ROI2vecs[ROI1]
            #     # region_vecs1_ = np.expand_dims(region_vecs1, 2)
            #     rxr = pb_outer(region_vecs.T, region_vecs1.T, flat=True)
            #     # region_x_region_l = []
            #     # for trial in range(n_trials):
            #     #     region_x_region = np.matmul(region_vecs_[:, trial],
            #     #                                 region_vecs1_[:, trial].T)
            #     #     region_x_region_vec = region_x_region.flatten()
            #     #     region_x_region_l.append(region_x_region_vec)
            #     # rxr = np.stack(region_x_region_l, axis=1).T
            #     RDM_rxr = np.corrcoef(rxr)
            #     # RDM_rxr = np.random.normal(0, 1, (n_trials, n_trials))
            #     del rxr
            #     z_rxr_obj = RDM_x_RDM(RDM_rxr, RDM_stim_obj)
            #     rxr_vecs['obj'].append(z_rxr_obj)
            #     z_rxr_scn = RDM_x_RDM(RDM_rxr, RDM_stim_scn)
            #     rxr_vecs['scn'].append(z_rxr_scn)
            #     z_rxr_dif = RDM_x_RDM(RDM_rxr, RDM_stim_dif)
            #     rxr_vecs['dif'].append(z_rxr_dif)
            # rxr_mats['obj'].append(rxr_vecs['obj'])
            # rxr_mats['scn'].append(rxr_vecs['scn'])
            # rxr_mats['dif'].append(rxr_vecs['dif'])

        triple_z['obj'].append(calculate_triple_connectivity(ROI_to_RDM_fMRI,
                                                             RDM_stim_obj,
                                                             ROIs))
        triple_z['scn'].append(calculate_triple_connectivity(ROI_to_RDM_fMRI,
                                                             RDM_stim_scn,
                                                             ROIs))
        triple_z['dif'].append(calculate_triple_connectivity(ROI_to_RDM_fMRI,
                                                             RDM_stim_dif,
                                                             ROIs))
        rxr['obj'].append(np.array(rxr_mats['obj']))
        rxr['scn'].append(np.array(rxr_mats['scn']))
        rxr['dif'].append(np.array(rxr_mats['dif']))

    for ROI in ROI_to_dif:
        try:
            # catch NaNs
            zs_dif_ = regress_out(ROI_to_obj[ROI], ROI_to_dif[ROI])
            zs_dif_ = regress_out(ROI_to_scn[ROI], zs_dif_)
            ROI_to_dif_[ROI] = zs_dif_
        except ValueError:
            pass

        ROI_to_IRAF_obj[ROI] = np.array(ROI_to_IRAF_obj[ROI])
        test = np.max(ROI_to_IRAF_obj[ROI])
        # if test > 1:
        #     print(f'{test} | {ROI=}')
        #     quit()
        ROI_to_IRAF_scn[ROI] = np.array(ROI_to_IRAF_scn[ROI])
        ROI_to_IRAF_dif[ROI] = np.array(ROI_to_IRAF_dif[ROI])
        ROI_to_IRAF_dif_[ROI] = np.zeros(ROI_to_IRAF_dif[ROI].shape)
        n_trials = ROI_to_IRAF_obj[ROI].shape[1]
        for trial_i in range(n_trials):
            IRAF_dif_ = regress_out(ROI_to_IRAF_obj[ROI][:, trial_i],
                                    ROI_to_IRAF_dif[ROI][:, trial_i])
            # IRAF_dif_ = ROI_to_IRAF_dif[ROI][:, trial_i]
            # print(f'{len(IRAF_dif_)=}')
            # print(f'{IRAF_dif_=}')
            # quit()
            IRAF_dif_ = regress_out(ROI_to_IRAF_scn[ROI][:, trial_i],
                                    IRAF_dif_)
            ROI_to_IRAF_dif_[ROI][:, trial_i] = IRAF_dif_

    # IRAFs_con = np.corrcoef(IRAFs_obj)
    # IRAFs_con[np.isinf(IRAFs_con)] = np.nan
    # IRAFs_con = np.arctanh(IRAFs_con)
    # IRAFs_connectivity_all.append(IRAFs_con)
    dir_out = r'result_pics/RSA_conn'
    age_str = 'healthy' if age == 'healthy' else 'YA' if age == 1 else 'OA'
    cin_str = '' if cin is None else \
        '_Con' if cin == 1 else \
        '_Inc' if cin == 2 else '_Neu'
    sem_str = '_sem' if semantic else ''
    el_str = '' if semantic else '_early' if early else '_late'
    IRAFs_connectivity_obj = get_IRAF_connectivity_matrix(ROI_to_IRAF_obj,
                                                          n_ROIs, ROIs)
    # fp_obj = fr'{dir_out}/{age_str}/obj{cin_str}{sem_str}{el_str}.png'
    # plot_connectivity(IRAFs_connectivity_obj, ticks, tick_labels, tick_lows,
    #                   title=f'Object, {age=}, {cin=}, {early=}, {semantic=}',
    #                   fp=fp_obj)

    IRAFs_connectivity_scn = get_IRAF_connectivity_matrix(ROI_to_IRAF_scn,
                                                          n_ROIs, ROIs)
    # fp_scn = fr'{dir_out}/{age_str}/scene{cin_str}{sem_str}{el_str}.png'
    # plot_connectivity(IRAFs_connectivity_scn, ticks, tick_labels, tick_lows,
    #                   title=f'Scene, {age=}, {cin=}, {early=}, {semantic=}',
    #                   fp=fp_scn)

    IRAFs_connectivity_dif = get_IRAF_connectivity_matrix(ROI_to_IRAF_dif,
                                                          n_ROIs, ROIs)
    # fp_dif = fr'{dir_out}/{age_str}/dif{cin_str}{sem_str}{el_str}.png'
    # plot_connectivity(IRAFs_connectivity_dif, ticks, tick_labels, tick_lows,
    #                   title=f'Dif, {age=}, {cin=}, {early=}, {semantic=}',
    #                   fp=fp_dif)


    IRAFs_connectivity_dif_ = get_IRAF_connectivity_matrix(ROI_to_IRAF_dif_,
                                                          n_ROIs, ROIs)
    # fp_dif = fr'{dir_out}/{age_str}/dif_regressed_{cin_str}{sem_str}{el_str}.png'
    # plot_connectivity(IRAFs_connectivity_dif_, ticks, tick_labels, tick_lows,
    #                   title=f'Dif regressed, {age=}, {cin=}, {early=}, {semantic=}',
    #                   fp=fp_dif)


    fig, axs = plt.subplots(1, 4, figsize=(27, 7))
    plot_connectivity(IRAFs_connectivity_obj, ticks, tick_labels, tick_lows,
                      title=f'obj {age=}, {cin=}, {early=}, {semantic=}',
                      ax=axs[0], t=False)
    # plot_connectivity(IRAFs_connectivity_obj, ticks, tick_labels, tick_lows,
    #                   title=f'obj t-value {age=}, {cin=}, {early=}, {semantic=}',
    #                   ax=axs[1], t=True)
    plot_connectivity(IRAFs_connectivity_scn, ticks, tick_labels, tick_lows,
                      title=f'scene {age=}, {cin=}, {early=}, {semantic=}', ax=axs[1])
    plot_connectivity(IRAFs_connectivity_dif, ticks, tick_labels, tick_lows,
                      title=f'dif {age=}, {cin=}, {early=}, {semantic=}', ax=axs[2])
    plot_connectivity(IRAFs_connectivity_dif_, ticks, tick_labels, tick_lows,
                      title=f'dif regressed {age=}, {cin=}, {early=}, {semantic=}',
                      ax=axs[3])
    plt.tight_layout()
    fp_subplots = fr'{dir_out}/{age_str}/all{cin_str}{sem_str}{el_str}.png'
    plt.savefig(fp_subplots)
    # plt.show()
    # quit()

    d_out = {
            #'rxr': rxr,
            'triple_z': triple_z, # triple_z is already a dict
            'IRAF_conn': {'obj': IRAFs_connectivity_obj,
                           'scn': IRAFs_connectivity_scn,
                           'dif': IRAFs_connectivity_dif,
                           'dif_': IRAFs_connectivity_dif_},
            'activity': ROI_to_activity,
             'IRAFs_ROI': {'obj': ROI_to_IRAF_obj,
                           'scn': ROI_to_IRAF_scn,
                           'dif': ROI_to_IRAF_dif,
                           'dif_': ROI_to_IRAF_dif_},
             'z': {'obj': ROI_to_obj,
                   'scn': ROI_to_scn,
                   'dif': ROI_to_dif,
                   'dif_': ROI_to_dif_}}
    for key, d_sub in d_out.items():
        for key2, d_sub_sub in d_sub.items():
            if isinstance(d_sub_sub, defaultdict):
                d_out[key][key2] = dict(d_sub_sub)
    return d_out



def plot_connectivity(conn, ticks, tick_labels, tick_lows, title='', fp=None,
                      ax=None, t=False, no_avg=False):
    if not no_avg:
        M_connect = np.nanmean(conn, axis=0)
        if t:
            M_connect = M_connect / np.nanstd(conn, axis=0) * np.sqrt(len(conn))
    else:
        M_connect = conn
    # M_connect = np.nanmedian(conn, axis=0)
    vmin = np.nanquantile(M_connect, .01)
    vmax = np.nanquantile(M_connect, .99)
    print(f'vmin: {vmin}, vmax: {vmax}')
    # vmin = .2
    # vmax = .8
    if ax is None:
        plt.figure(figsize=(10, 10))
    else:
        plt.sca(ax)
    plt.title(title)
    plt.imshow(M_connect, vmin=vmin, vmax=vmax, cmap='turbo')
    plt.yticks(ticks, tick_labels, fontsize=8)
    plt.xticks(ticks, tick_labels, fontsize=8, rotation=90)
    for low in tick_lows:
        plt.plot([0, M_connect.shape[0]], [low, low], 'w', linewidth=0.5)
        plt.plot([low, low], [0, M_connect.shape[0]], 'w', linewidth=0.5)
    plt.xlim([0, M_connect.shape[0]])
    plt.ylim([0, M_connect.shape[0]])
    plt.colorbar(shrink=0.7, aspect=20*0.7)
    if fp is not None:
        Path(fp).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fp)
    if ax is None:
        plt.show()

def run_multi_settings():
    # semantic = False
    for early, semantic in [(True, False), (False, False), (False, True)]:
        for age in [1, 2, 'healthy', ]:
            # age = 'healthy'
            for cin in [None, 1, 2, 3]:
                age_str = 'healthy' if age == 'healthy' else \
                    'YA' if age == 1 else 'OA'
                cin_str = '' if cin is None else \
                    '_Con' if cin == 1 else \
                        '_Inc' if cin == 2 else '_Neu'
                sem_str = '_sem' if semantic else ''
                el_str = '' if semantic else '_early' if early else '_late'
                fp_out = fr'cache/RSA/{age_str}{cin_str}{sem_str}{el_str}.pkl'
                Path(fp_out).parent.mkdir(parents=True, exist_ok=True)
                d = pickle_wrap(fp_out,
                                lambda: mass_RDM_x_RDM(age=age, cin=cin,
                                                       early=early),
                                easy_override=True)

def pb_outer(a, b, flat=False):
    a = np.array(a)
    b = np.array(b)
    a = np.expand_dims(a, axis=2)
    b = np.expand_dims(b, axis=1)
    c = a * b
    if flat:
        c = c.reshape(-1, c.shape[1]*c.shape[2])
    return c

    # # print(b.shape)
    # a = np.expand_dims(a, axis=2)
    # # print(a.shape)
    # # print(a)
    # a = np.repeat(a, b.shape[1], axis=2)
    # # print(a)
    # # print(a.shape)
    # # quit()
    # b = np.expand_dims(b, axis=1)
    # # print(b)
    # b = np.repeat(b, a.shape[1], axis=1)
    # # print(a.shape)
    # c = a * b
    # if flat:
    #     c = c.reshape(-1, c.shape[1]*c.shape[2])
    # # print(c.shape)
    # return c

# a = [[1, 2, 1], [3, 4, 5]]
# b = [[4, 5, 1], [6, 7, 6]]
# a = np.array(a)
# b = np.array(b)
# a = np.expand_dims(a, axis=2)
# print(a.shape)
# b = np.expand_dims(b, axis=1)
# print(b.shape)
# a = np.array([[1, 2])
# b = np.array([[[3, 4]])
# print(a * b)
# quit()

# c = np.outer(a, b)
# # print(c)
# c = pb_out/er(a, b)
# print(c)
# print(c.shape)
# quit()

if __name__ == '__main__':
    # TODO: 3 way correlation, Region A RDM x Region B RDM x Stimulus RDM
    run_multi_settings()

