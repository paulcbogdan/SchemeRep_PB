from pickle_wrap import pickle_wrap
from collections import defaultdict

import numpy as np

from DNN_vectors import get_DNN_vecs, get_img_fns
from ROIs import add_ROI_info, get_BN_and_resample, get_combined_BNA
from organize_bhv import get_trial_info, get_all_sns
from nilearn import image

from utils import stdize, nan_ar, defaultdict_to_dict, pb_outer_double_multi
from wordvec_get_vectors import get_semantic_vectors
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
    if obj_only:
        all_vecs = vecs_obj
    elif scene_only:
        all_vecs = vecs_scene
    else:
        all_vecs = abs(vecs_obj - vecs_scene)
    RDM_stim = np.corrcoef(all_vecs)
    pd.DataFrame(RDM_stim).to_csv('RDM_stim_mine.csv')
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


def RDM_x_RDM(fMRI_RDM, stim_RDM):
    assert fMRI_RDM.shape == stim_RDM.shape, 'RDMs must be the same shape: ' \
       f'fMRI_RDM.shape = {fMRI_RDM.shape}, stim_RDM.shape = {stim_RDM.shape}'
    tril_idx = np.tril_indices_from(fMRI_RDM, k=-1)
    fMRI_vec = fMRI_RDM[tril_idx]
    stim_vec = stim_RDM[tril_idx]
    r, _ = stats.spearmanr(fMRI_vec, stim_vec)
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
    return np.array(IRAFs)


def get_IRAF_connectivity_matrix(ROI_to_IRAF, n_ROIs, ROIs):
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
        mat = np.corrcoef(ar)
        mat[np.diag_indices_from(mat)] = 0
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if i % 2 == j % 2:
                    if (i < mat.shape[0] - 1):
                        mat[i, j] = mat[i+1, j]
        mat[mat > .99] = .99
        mat[mat < -.99] = -.99
        mat = np.arctanh(mat)
        all_matricies.append(mat)
    # quit()
    return all_matricies

def get_triple_connectivity(ROI_to_RDM_fMRI, RDM_stim, ROIs):
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

def get_ROI_vecs(ROIs, ROI_nums, atlas, img, nan_thresh=.25):
    ROI2vecs = {}
    ROI2vecs_down = {}
    for j, (ROI, ROI_num) in enumerate(zip(ROIs, ROI_nums)):
        atlas_roi = atlas['maps'].get_fdata() == ROI_num
        atlas_roi_downsample = np.copy(atlas_roi)
        atlas_roi_downsample[::3, :, :] = False
        atlas_roi_downsample[:, ::3, :] = False
        atlas_roi_downsample[:, :, ::3] = False

        region_vecs = img[atlas_roi]
        region_vecs_down = img[atlas_roi_downsample]
        voxels_w_nan = np.isnan(region_vecs).any(axis=1)
        n_nans_ROI = np.sum(voxels_w_nan)
        if n_nans_ROI / len(voxels_w_nan) > nan_thresh:  # more than 10%
            continue
        region_vecs = region_vecs[~voxels_w_nan, :]
        ROI2vecs[ROI] = region_vecs
        region_vecs_down = (region_vecs_down.T -
                            np.nanmean(region_vecs_down, axis=1)) / \
                           np.nanstd(region_vecs_down, axis=1)
        ROI2vecs_down[ROI] = region_vecs_down
    return ROI2vecs, ROI2vecs_down

def ROI_dict2ar(ROI2vec, ROIs, n_trials):
    ROI_vec_down_ar = []
    for ROI in ROIs:
        if ROI not in ROI2vec:
            blank = np.full((n_trials, 1), np.nan)
            ROI_vec_down_ar.append(blank)
        else:
            ROI_vec_down_ar.append(ROI2vec[ROI])
    return ROI_vec_down_ar

def run_rxr(ROIs, ROI_nums, ROI2vecs_down, j, inc, rxr_all_products,
           RDM_stims, rxr_mats):
    for k, (ROI1, ROI_num1) in enumerate(zip(ROIs, ROI_nums)):
        if ROI1 not in ROI2vecs_down:
            continue
        if k <= j:
            continue
        rxr = rxr_all_products[j % inc, k]
        nans = np.isnan(rxr[0])
        rxr = rxr[:, ~nans]
        RDM_rxr = np.corrcoef(rxr)
        for key in RDM_stims:
            rxr_mats[key][j, k] = RDM_x_RDM(RDM_rxr, RDM_stims['obj'])

def get_mean_activity(region_vecs, sort_by):
    M = np.nanmean(region_vecs, axis=0)
    M_sorted = []
    for a, _ in sorted(zip(M, sort_by), key=lambda x: x[1]):
        M_sorted.append(a)
    return M_sorted

def analyze_subj(sn, cin, d_vecs, n_ROIs, ROIs, ROI_nums, atlas, n_trials,
                 ROI_to_z, ROI_to_IRAF, triple_z, ROI_to_activity, rxr_all,
                 do_rxr):
    print(f'Onto: {sn}')
    df_sn = get_trial_info(sn)
    if cin is not None:
        df_sn = df_sn[df_sn['CIN'] == cin]
    RDM_stims = {'obj': get_stim_RDM(df_sn, d_vecs, obj_only=True),
                 'scn': get_stim_RDM(df_sn, d_vecs, obj_only=False, scene_only=True),
                 'dif': get_stim_RDM(df_sn, d_vecs, obj_only=False, dif=True)}

    img = image.load_img(df_sn['fp_fMRI']).get_fdata()

    ROI_to_RDM_fMRI = {}
    rxr_mats = {'obj': nan_ar((n_ROIs, n_ROIs)),
                'scn': nan_ar((n_ROIs, n_ROIs)),
                'dif': nan_ar((n_ROIs, n_ROIs))}

    # NaN thresh happens here
    ROI2vecs, ROI2vecs_down = get_ROI_vecs(ROIs, ROI_nums, atlas, img)
    ROI_vec_down_ar = ROI_dict2ar(ROI2vecs_down, ROIs, n_trials)
    rxr_all_products = None
    for j, (ROI, ROI_num) in tqdm(enumerate(zip(ROIs, ROI_nums)),
                                  desc='looping ROIs outer',
                                  total=len(ROIs), leave=True, ncols=80,
                                  position=0):
        if ROI not in ROI2vecs:
            continue
        region_vecs = ROI2vecs[ROI]
        ROI_to_activity[ROI].append(get_mean_activity(region_vecs, df_sn['obj']))
        RDM_fMRI = np.corrcoef(region_vecs.T)
        ROI_to_RDM_fMRI[ROI] = RDM_fMRI
        for key in ['obj', 'scn', 'dif']:
            ROI_to_z[key][ROI].append(RDM_x_RDM(RDM_fMRI, RDM_stims[key]))
            IRAFs = get_IRAFs(RDM_fMRI, RDM_stims[key], df_sn)

            ar = np.append(ROI_to_IRAF[key][ROI], IRAFs[None, :], axis=0)
            ROI_to_IRAF[key][ROI] = ar
        if do_rxr:
            inc = 3
            if j % inc == 0:
                rxr_all_products = pb_outer_double_multi(
                    ROI_vec_down_ar[j:j + inc], ROI_vec_down_ar, flat=True)
            run_rxr(ROIs, ROI_nums, ROI2vecs_down, j, inc, rxr_all_products,
                    RDM_stims, rxr_mats)

    if do_rxr:
        del rxr_all_products
        for key in ['obj', 'scn', 'dif']:
            rxr_all[key].append(np.array(rxr_mats[key]))

    for key in ['obj', 'scn', 'dif']:
        triple_z[key].append(get_triple_connectivity(ROI_to_RDM_fMRI,
                                                     RDM_stims[key], ROIs))




def mass_RDM_x_RDM(age=1, cin=None, semantic=False, early=False, do_rxr=False,
                   bilateral=False, combine_regions=False):
    if semantic:
        d_vecs = get_semantic_vectors()
    else:
        d_vecs = get_DNN_vecs(early=early, PCA=True)
    # ROIs, ROI_nums, ticks, tick_labels, tick_lows, n_ROIs = add_ROI_info()
    if combine_regions:
        atlas = get_combined_BNA(combine_bilateral=bilateral)
    else:
        atlas = get_BN_and_resample(combine_bilateral=bilateral)
    age2sn = get_all_sns()
    n_trials = 114 if cin is None else 38

    ROI_to_z = defaultdict(lambda: defaultdict(list))
    ROI_to_IRAF = defaultdict(lambda: defaultdict(lambda: np.full((0, n_trials),
                                                                  np.nan)))
    ROI_to_activity = defaultdict(list)
    triple_z = {'obj': [], 'scn': [], 'dif': []}
    rxr_all = {'obj': [], 'scn': [], 'dif': []}

    for i, sn in tqdm(enumerate(age2sn[age]),
                      desc=f'Looping subjects: age2sn[{age}]'):
        analyze_subj(sn, cin, d_vecs, atlas['n_ROIs'], atlas['ROIs'],
                     atlas['ROI_nums'], atlas, n_trials,
                     ROI_to_z, ROI_to_IRAF, triple_z, ROI_to_activity, rxr_all,
                     do_rxr)

    for ROI in atlas['ROIs']:
        ROI_to_z['dif_'][ROI] = utils.regress_out_multi([ROI_to_z['obj'][ROI],
                                                         ROI_to_z['scn'][ROI]],
                                                        ROI_to_z['dif'][ROI])
        ROI_to_IRAF['dif_'][ROI] = np.full(ROI_to_IRAF['dif'][ROI].shape, np.nan)
        for trial_i in range(n_trials):
            IRAF_dif_ = utils.regress_out_multi([ROI_to_IRAF['obj'][ROI][:, trial_i],
                                                 ROI_to_IRAF['scn'][ROI][:, trial_i]],
                                                ROI_to_IRAF['dif'][ROI][:, trial_i])
            ROI_to_IRAF['dif_'][ROI][:, trial_i] = IRAF_dif_

    IRAF_conn = {}
    for key in ['obj', 'scn', 'dif', 'dif_']:
        IRAF_conn[key] = get_IRAF_connectivity_matrix(ROI_to_IRAF[key],
                                                      atlas['n_ROIs'],
                                                      atlas['ROIs'])

    d_out = {'rxr': rxr_all,
            'triple_z': triple_z,
            'IRAF_conn': IRAF_conn,
            'activity': ROI_to_activity,
            'IRAFs_ROI': ROI_to_IRAF,
            'z': ROI_to_z}
    d_out = defaultdict_to_dict(d_out)
    return d_out


def run_multi_settings():
    # semantic = False
    rxr = False
    combine_regions = False
    bilateral = True
    for early, semantic in [(True, False), (False, False), (False, True)]:
            # age = 'healthy'
        for cin in [None, 1, 2, 3]:
            for age in [1, 2, 'healthy', ]:
                age_str = 'healthy' if age == 'healthy' else \
                    'YA' if age == 1 else 'OA'
                cin_str = '' if cin is None else \
                    '_Con' if cin == 1 else \
                        '_Inc' if cin == 2 else '_Neu'
                sem_str = '_sem' if semantic else ''
                el_str = '' if semantic else '_early' if early else '_late'
                rxr_str = '_rxr' if rxr else ''
                combine_str = '_comb' if combine_regions else ''
                bilat_str = '_bil' if bilateral else ''
                fp_out = fr'cache/RSA/{age_str}{cin_str}{sem_str}{el_str}' \
                         fr'{combine_str}{bilat_str}{rxr_str}.pkl'
                Path(fp_out).parent.mkdir(parents=True, exist_ok=True)
                d = pickle_wrap(fp_out,
                                lambda: mass_RDM_x_RDM(age=age, cin=cin,
                                                       early=early,
                                                       do_rxr=rxr,
                                                       semantic=semantic,
                                                       bilateral=bilateral,
                                               combine_regions=combine_regions),
                                easy_override=False)


if __name__ == '__main__':
    # TODO: 3 way correlation, Region A RDM x Region B RDM x Stimulus RDM
    run_multi_settings()

