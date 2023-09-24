from pickle_wrap import pickle_wrap
from collections import defaultdict

import numpy as np

from DNN_vectors import get_DNN_vecs
from ROIs import get_BN_and_resample, get_combined_BNA
from organize_bhv import get_trial_info, get_all_sns
from nilearn import image

from stim_vec import get_stim_RDM
from utils import stdize, nan_ar, defaultdict_to_dict, pb_outer_double_multi
from wordvec_get_vectors import get_semantic_vectors
import utils
import scipy.stats as stats

from tqdm import tqdm
from pathlib import Path


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
        # print(fMRI_RDM[i, :])
        stim_vec_std = np.delete(stim_RDM[i, :], i)
        # print(stim_RDM[i, :])
        # print()
        # fMRI_vec_std = stdize(fMRI_vec_std) # exclude correlation w itself
        # stim_vec_std = stdize(stim_vec_std)
        # TODO: maybe use spearna?
        # r = (fMRI_vec_std @ stim_vec_std) / len(fMRI_vec_std)
        # r, _ = stats.pearsonr(fMRI_vec_std, stim_vec_std)
        r, _ = stats.spearmanr(fMRI_vec_std, stim_vec_std)
        IRAFs.append(r)
    IRAFs = [IRAF for (IRAF, _) in sorted(zip(IRAFs, df_sn['obj']),
                                          key=lambda x: x[1])]
    return np.array(IRAFs)


def get_IRAF_connectivity_matrix(ROI_to_IRAF, n_ROIs, ROIs):
    first_key = next(iter(ROI_to_IRAF.keys()))
    subj_timeseries_shape = (n_ROIs, ROI_to_IRAF[first_key].shape[1])
    subj2timeseries = defaultdict(lambda: np.full(subj_timeseries_shape,
                                                  np.nan))
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

def get_ROI_vecs(sn, ROIs, ROI_nums, atlas, img, ROI_regions,
                 fp_fMRI_col, nan_thresh=.25, vec_prod=False, org_by_region=False):
    n_ROIs = len(ROIs)
    n_regions = len(np.unique(ROI_regions))
    vec_prod_str = '_vp' if vec_prod else ''
    nan_str = f'_nan{nan_thresh}' if nan_thresh != .25 else ''
    org_by_region_str = '_oByR' if org_by_region else ''
    fp_cache = fr'cache\ROI2vecs\sn{sn}_{fp_fMRI_col}_nROI{n_ROIs}_reg{n_regions}' \
               fr'{vec_prod_str}{org_by_region_str}{nan_str}.pkl'
    f = lambda: get_ROI_vecs_(ROIs, ROI_nums, atlas, img, ROI_regions,
                  nan_thresh=nan_thresh, vec_prod=vec_prod,
                  org_by_region=org_by_region)
    r2vecs, r2vecs_down = pickle_wrap(fp_cache, f, verbose=True)
    return r2vecs, r2vecs_down


def get_ROI_vecs_(ROIs, ROI_nums, atlas, img, ROI_regions,
                  nan_thresh=.25, vec_prod=False, org_by_region=False):
    ROI2vecs = {}
    ROI2vecs_down = {}
    region2vecs = defaultdict(list)
    idxs = np.arange(114)
    # np.random.shuffle(idxs)
    for j, (ROI, ROI_num, region) in enumerate(zip(ROIs, ROI_nums, ROI_regions)):
        atlas_roi = atlas['maps'].get_fdata() == ROI_num
        # atlas_roi_downsample = np.copy(atlas_roi)
        # atlas_roi_downsample[::3, :, :] = False
        # atlas_roi_downsample[:, ::3, :] = False
        # atlas_roi_downsample[:, :, ::3] = False

        region_vecs = img[atlas_roi]
        region_vecs = region_vecs[:, idxs]
        # print(region_vecs.shape)
        region_vecs = np.random.normal(size=region_vecs.shape)
        # print(region_vecs.shape)

        # print(region_vecs)
        # quit()
        # region_vecs_down = img[atlas_roi_downsample]
        voxels_w_nan = np.isnan(region_vecs).any(axis=1)
        n_nans_ROI = np.sum(voxels_w_nan)
        if n_nans_ROI / len(voxels_w_nan) > nan_thresh:  # more than 10%
            continue
        region_vecs = region_vecs[~voxels_w_nan, :]
        # down_voxels_w_nan = np.isnan(region_vecs_down).any(axis=1)
        # region_vecs_down = region_vecs_down[~down_voxels_w_nan, :]
        region_vecs = region_vecs.T
        ROI2vecs[ROI] = region_vecs
        # region_vecs_down = (region_vecs_down.T -
        #                     np.mean(region_vecs_down, axis=1)) / \
        #                    np.std(region_vecs_down, axis=1)
        # ROI2vecs_down[ROI] = region_vecs_down
        if org_by_region:
            # print(f'{region} | {ROI}')
            region2vecs[region].append(np.nanmean(region_vecs, axis=1))
    # quit()
    region2vecs = dict(region2vecs)
    for region, l in region2vecs.items():
        region2vecs[region] = np.array(l).T

    if org_by_region:
        return region2vecs, region2vecs
    elif vec_prod:
        return get_ROI_vecs_prod(ROI2vecs_down)
    else:
        return ROI2vecs, ROI2vecs_down

def get_ROI_vecs_prod(ROI2vecs):
    ROI2vec_prods = {}
    for ROI, vecs in ROI2vecs.items():
        vecs_ = stdize(vecs, axis=1, nans=True)
        ROI2vec_prods[ROI] = utils.pb_outer(vecs_, vecs_, flat=False)
        idxs = np.tril_indices_from(ROI2vec_prods[ROI][0, :], k=-1)
        ROI2vec_prods[ROI] = ROI2vec_prods[ROI][:, idxs[0], idxs[1]]
        # print('n nans = ', np.sum(np.isnan(ROI2vec_prods[ROI])),
        #       '| prev = ', np.sum(np.isnan(vecs)),
        #       'total = ', ROI2vec_prods[ROI].size)
    return ROI2vec_prods, None


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
    M = np.nanmean(region_vecs, axis=1)
    M_sorted = []
    for a, _ in sorted(zip(M, sort_by), key=lambda x: x[1]):
        M_sorted.append(a)
    return np.array(M_sorted)

def analyze_subj(sn, cin, d_vecs, n_ROIs, ROIs, ROI_nums, atlas, n_trials,
                 ROI_to_z, ROI_to_IRAF, triple_z, ROI_to_activity, rxr_all,
                 do_rxr, ROI_regions, bhv, stim_keys,
                 fp_fMRI_col='fp_fMRI',
                 org_by_region=False, region_vec_prod=True):
    assert (not do_rxr) or (not region_vec_prod), \
        f'Cannot do both rxr ({do_rxr=}) and region vec prod ({region_vec_prod=})'

    print(f'Onto: {sn}')
    df_sn = get_trial_info(sn)

    if cin is not None:
        df_sn = df_sn[df_sn['CIN'] == cin]

    df_sn_ = df_sn.sort_values(by='obj') # TODO: makes more sense to do this sort at the start
    bhv_cols = ['trial', 'run', 'fp_fMRI', 'obj', 'scene', 'obj_rename',
                'scene_rename', 'CIN', 'perceived_con', 'ON', 'hit_bool']
    for key in bhv_cols:
        try:
            bhv[key].append(df_sn_[key].values)
        except KeyError:
            bhv[key].append(np.full(len(df_sn_), np.nan))

    RDM_stims = {'obj': get_stim_RDM(df_sn, d_vecs, obj_only=True),
                 'obj_abs': get_stim_RDM(df_sn, d_vecs, obj_only=True, take_abs=True),
                 'scn': get_stim_RDM(df_sn, d_vecs, scene_only=True),
                 'scn_abs': get_stim_RDM(df_sn, d_vecs, scene_only=True, take_abs=True),
                 'dif': get_stim_RDM(df_sn, d_vecs, dif=True),
                 'dif_abs': get_stim_RDM(df_sn, d_vecs, dif=True, take_abs=True),
                 'prd': get_stim_RDM(df_sn, d_vecs, prod=True),
                 'prd_abs': get_stim_RDM(df_sn, d_vecs, prod=True, take_abs=True),
                 'add': get_stim_RDM(df_sn, d_vecs, add=True),
                 'add_abs': get_stim_RDM(df_sn, d_vecs, add=True, take_abs=True),
                 }
    assert len(RDM_stims) == len(stim_keys), f'{len(RDM_stims)=} != {len(stim_keys)=}'


    img = image.load_img(df_sn[fp_fMRI_col]).get_fdata()

    ROI_to_RDM_fMRI = {}
    rxr_mats = {}
    for key in stim_keys:
        rxr_mats[key] = nan_ar((n_ROIs, n_ROIs))

    # NaN thresh happens here
    ROI2vecs, ROI2vecs_down = get_ROI_vecs(sn, ROIs, ROI_nums, atlas, img,
                                            ROI_regions, fp_fMRI_col,
                                            vec_prod=region_vec_prod,
                                            org_by_region=org_by_region)

    if org_by_region:
        ROIs = atlas['tick_labels']

    if do_rxr:
        ROI_vec_down_ar = ROI_dict2ar(ROI2vecs_down, ROIs, n_trials)
    rxr_all_products = None
    for j, (ROI, ROI_num) in tqdm(enumerate(zip(ROIs, ROI_nums)),
                                  desc='looping ROIs outer',
                                  total=len(ROIs), leave=True, ncols=80,
                                  position=0):
        if ROI not in ROI2vecs:
            nan_fill = np.full((1, n_trials), np.nan)
            ROI_to_activity[ROI] = np.append(ROI_to_activity[ROI], nan_fill,
                                             axis=0)
            for key in stim_keys:
                ROI_to_z[key][ROI].append(np.nan)
                nan_fill = np.full((1, n_trials), np.nan)
                ROI_to_IRAF[key][ROI] = np.append(ROI_to_IRAF[key][ROI],
                                                  nan_fill, axis=0)
            ROI_to_RDM_fMRI[ROI] = nan_ar((n_trials, n_trials))
            continue
        region_vecs = ROI2vecs[ROI]
        activity =  get_mean_activity(region_vecs, df_sn['obj'])[None, :]
        ROI_to_activity[ROI] = np.append(ROI_to_activity[ROI], activity, axis=0)
        RDM_fMRI = np.corrcoef(region_vecs)
        ROI_to_RDM_fMRI[ROI] = RDM_fMRI
        # print(RDM_fMRI.shape)
        # quit()
        for key in stim_keys:
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
        for key in stim_keys:
            rxr_all[key].append(np.array(rxr_mats[key]))

    for key in stim_keys:
        triple_z[key].append(get_triple_connectivity(ROI_to_RDM_fMRI,
                                                     RDM_stims[key], ROIs))




def mass_RDM_x_RDM(age=1, cin=None, semantic=False, early=False, DNN_layer=2,
                   do_rxr=False,
                   bilateral=False, combine_regions=False, vec_prod=False,
                   org_by_region=False, fp_fMRI_col='fp_fMRI'):
    if semantic:
        d_vecs = get_semantic_vectors()
    else:
        d_vecs = get_DNN_vecs(DNN_layer=DNN_layer, PCA=True)
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
    ROI_to_activity = defaultdict(lambda: np.full((0, n_trials), np.nan))
    stim_keys = ['obj', 'obj_abs',
                 'scn', 'scn_abs',
                 'dif', 'dif_abs',
                 'prd', 'prd_abs',
                 'add', 'add_abs']
    triple_z = {}
    rxr_all = {}
    for key in stim_keys:
        triple_z[key] = []
        rxr_all[key] = []

    if org_by_region:
        atlas['n_ROIs'] = len(atlas['tick_labels'])

    bhv = defaultdict(list)
    sns = []
    for i, sn in tqdm(enumerate(age2sn[age]),
                      desc=f'Looping subjects: age2sn[{age}]'):
        analyze_subj(sn, cin, d_vecs, atlas['n_ROIs'], atlas['ROIs'],
                     atlas['ROI_nums'], atlas, n_trials,
                     ROI_to_z, ROI_to_IRAF, triple_z, ROI_to_activity, rxr_all,
                     do_rxr, atlas['ROI_regions'], bhv, stim_keys,
                     region_vec_prod=vec_prod,
                     org_by_region=org_by_region,
                     fp_fMRI_col=fp_fMRI_col,
                     )
        sns.append(sn)



    ROIs = atlas['tick_labels'] if org_by_region else atlas['ROIs']
    keys_sans_obj_scn = [key for key in stim_keys if
                         'obj' != key and 'scn' != key]
    for ROI in ROIs:
        for key in keys_sans_obj_scn:
            key_mod = f'{key}_'
            ROI_to_z[key_mod][ROI] = utils.regress_out_multi([ROI_to_z['obj'][ROI],
                                                             ROI_to_z['scn'][ROI]],
                                                            ROI_to_z[key][ROI])
            ROI_to_IRAF[key_mod][ROI] = np.full(ROI_to_IRAF[key][ROI].shape, np.nan)
            for trial_i in range(n_trials):
                IRAF_dif_ = utils.regress_out_multi([ROI_to_IRAF['obj'][ROI][:, trial_i],
                                                     ROI_to_IRAF['scn'][ROI][:, trial_i]],
                                                    ROI_to_IRAF[key][ROI][:, trial_i])
                ROI_to_IRAF[key_mod][ROI][:, trial_i] = IRAF_dif_

    IRAF_conn = {}
    for key in ROI_to_IRAF.keys():
        IRAF_conn[key] = get_IRAF_connectivity_matrix(ROI_to_IRAF[key],
                                                      atlas['n_ROIs'],
                                                      atlas['ROIs'])

    d_out = {'rxr': rxr_all,
            'triple_z': triple_z,
            'IRAF_conn': IRAF_conn,
            'activity': ROI_to_activity,
            'IRAFs_ROI': ROI_to_IRAF,
            'z': ROI_to_z,
            'bhv': bhv,
            'sns': sns
             }
    d_out = defaultdict_to_dict(d_out)
    return d_out


def run_multi_settings():
    # semantic = False
    rxr = False
    combine_regions = False
    bilateral = False # combines bilateral ROIs/regions
    vec_prod = False
    org_by_region = False
    assert not (org_by_region and combine_regions), \
        'Cannot combine regions and organize by region'
    fp_fMRI_col = 'obj_fMRI' # maps onto columns defined in get_trial_info
    # for cin in [None, 1, 2, 3]:
    # fp_fMRI_col = 'scn_fMRI'
    cin = None
    if True:
        if ['obj_fMRI', 'scn_fMRI', ]:
            for age in [1, 2, ]:
                # for early, semantic in [(True, False), (False, False), (False, True)]:
                for DNN_layer, semantic in [(2, False), (6, False), (4, False),
                                            (False, True), ]: # (True, False),
                # for early, semantic in [(True, False), (False, True), ]: # (True, False),
                    age_str = 'healthy' if age == 'healthy' else \
                        'YA' if age == 1 else 'OA'
                    cin_str = '' if cin is None else \
                        '_Con' if cin == 1 else \
                            '_Inc' if cin == 2 else '_Neu'
                    sem_str = '_sem' if semantic else ''
                    dnn_str = '' if semantic else \
                        '_late' if DNN_layer == -1 else \
                        '_early' if DNN_layer == 2 else \
                        f'_dnn{DNN_layer}'
                    rxr_str = '_rxr' if rxr else ''
                    combine_str = '_comb' if combine_regions else ''
                    bilat_str = '_bil' if bilateral else ''
                    vecprod_str = '_vecprod' if vec_prod else ''
                    by_region_str = '_byR' if org_by_region else ''
                    fp_in_str = fp_fMRI_col.replace('fMRI', '')
                    # fp_out = fr'cache/RSA/{age_str}{cin_str}{sem_str}{el_str}' \
                    #          fr'{combine_str}{bilat_str}{vecprod_str}{by_region_str}' \
                    #          fr'{rxr_str}.pkl'
                    fp_out = fr'cache/RSA/{fp_in_str}{age_str}{cin_str}{sem_str}{dnn_str}' \
                             fr'{combine_str}{bilat_str}{vecprod_str}{by_region_str}' \
                             fr'{rxr_str}.pkl'
                    Path(fp_out).parent.mkdir(parents=True, exist_ok=True)
                    f = lambda: mass_RDM_x_RDM(age=age, cin=cin,
                                               DNN_layer=DNN_layer, do_rxr=rxr,
                                               semantic=semantic,
                                               bilateral=bilateral,
                                               combine_regions=combine_regions,
                                               vec_prod=vec_prod,
                                               org_by_region=org_by_region,
                                               fp_fMRI_col=fp_fMRI_col
                                               )
                    d = pickle_wrap(fp_out, f, easy_override=True, verbose=True)


if __name__ == '__main__':
    # TODO: 3 way correlation, Region A RDM x Region B RDM x Stimulus RDM
    run_multi_settings()

