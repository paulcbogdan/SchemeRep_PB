import pickle

import matplotlib.pyplot as plt
import numpy as np

from atlas_utils import get_atlas
from fMRI_proc import get_ROI_vecs
from organize_bhv import get_trial_info
from old.plot_gen import plot_connectivity
from stim import get_stim_RDM, get_DNN_vecs, scipy_dist, prune_RSM_outliers
from utils import get_RSA_fn, tril_flat, stdize, pb_outer_euc, pb_outer
import scipy.stats as stats

import warnings
from time import time

warnings.filterwarnings('ignore', message='Mean of empty slice')
warnings.filterwarnings('ignore', message='Degrees of freedom <= 0 for slice.')


def roimap2np(ROI2ar, only_some=None):
    l = []
    for ROI, ar in ROI2ar.items():
        if only_some:
            for ROI_ in only_some:
                if ROI_ in ROI:
                    break
            else:
                continue
        l.append(ar)
    # print(np.array(l).shape)
    l = np.array(l)
    print(f'Data shape: {l.shape}')
    return l

def corr_matrix_last_two_dim(ar, nans=True, euc_dist=False,
                             stdize=True):

    # Second-to-last dim should be your variable
    # Last dim should be a time series
    m = np.nanmean if nans else np.mean
    # Creates nan in rs and rs_flat if every value in time series is identical
    if ar.shape[-1] > 1 and stdize:
        s = np.nanstd if nans else np.std
        M = np.expand_dims(m(ar, axis=-1), axis=-1)
        SD = np.expand_dims(s(ar, axis=-1), axis=-1)
        ar_std = (ar - M) / SD
        ar_std0 = np.expand_dims(ar_std, axis=-2)
        ar_std1 = np.expand_dims(ar_std, axis=-3)
    else:
        ar_std0 = np.expand_dims(ar, axis=-2)
        ar_std1 = np.expand_dims(ar, axis=-3)

    if euc_dist:
        # stdize can't actually make a difference because it applies to both
        #   ar_std0 and ar_std1
        rs = -abs(ar_std0 - ar_std1)
    else:
        rs = ar_std0 * ar_std1
    rs = m(rs, axis=-1)
    rs_flat = tril_flat(rs)
    return rs, rs_flat


def corr_last_dim(ar0, ar1, nans=True, euc_dist=False, stdize=True):
    # Last dim of both should be a time series
    # stdize by last dim for correlation
    m = np.nanmean if nans else np.mean
    if stdize:
        s = np.nanstd if nans else np.std
        M0 = np.expand_dims(m(ar0, axis=-1), axis=-1)
        SD0 = np.expand_dims(s(ar0, axis=-1), axis=-1)
        ar0_ = (ar0 - M0) / SD0
        M1 = np.expand_dims(m(ar1, axis=-1), axis=-1)
        SD1 = np.expand_dims(s(ar1, axis=-1), axis=-1)
        ar1_ = (ar1 - M1) / SD1
    else:
        ar0_ = ar0
        ar1_ = ar1
    if euc_dist:
        rs = -abs(ar0_ - ar1_)
    else:
        rs = ar0_ * ar1_
    rs = m(rs, axis=-1)
    return rs





def get_conn_vecs(vecs, vecs1=None, conn='euc'):
    vecs = stdize(vecs, axis=0, nans=True)
    if vecs1 is None:
        vecs1 = vecs
        cross = False
    else:
        vecs1 = stdize(vecs1, axis=0, nans=True)
        cross = True
    if conn == 'euc':
        vecs = pb_outer_euc(vecs, vecs1, tril=not cross,
                            flat=cross, nan_diag=not cross)
    elif conn == 'prod':
        vecs = pb_outer(vecs, vecs1,  tril=not cross,
                            flat=cross, nan_diag=not cross)
    else:
        raise ValueError(f'conn={conn} not recognized')
    return vecs

def calculate_cross_region_vecs(ROI2vecs, networks, conn='euc'):
    ROI2vecs_new = {}
    for network, ROIs in networks.items():
        if isinstance(ROIs, tuple):
            ROIs0 = ROIs[0]
            ROIs1 = ROIs[1]
            vecs0_l = []
            for ROI0_0 in ROIs0:
                vecs0_l.append(ROI2vecs[ROI0_0])
            vecs0 = np.concatenate(vecs0_l, axis=1)
            vecs1_l = []
            for ROI1_1 in ROIs1:
                vecs1_l.append(ROI2vecs[ROI1_1])
            vecs1 = np.concatenate(vecs1_l, axis=1)
            vecs = get_conn_vecs(vecs0, vecs1, conn=conn)
            # print(f'{vecs0.shape=}, {vecs1.shape=}, {vecs.shape=}')
            # quit()
        else:
            vecs_l = []
            for ROI0 in ROIs:
                vecs0 = ROI2vecs[ROI0]
                for ROI1 in ROIs:
                    if ROI0 == ROI1:
                        continue
                    vecs1 = ROI2vecs[ROI1]
                    vecs = get_conn_vecs(vecs0, vecs1, conn=conn)
                    vecs_l.append(vecs)
            vecs = np.concatenate(vecs_l, axis=1)
        assert vecs.shape[0] == 114, f'{vecs.shape=}'
        ROI2vecs_new[network] = vecs
    return ROI2vecs_new


def cluster_regions(ROI2vecs, networks):
    ROI2vecs_new = {}
    for network, ROIs in networks.items():
        vecs_l = []
        for ROI in ROIs:
            if ROI in ROI2vecs:
                vecs_l.append(ROI2vecs[ROI])
            elif f'{ROI}_L' in ROI2vecs:
                vecs_l.append(ROI2vecs[f'{ROI}_L'])
                vecs_l.append(ROI2vecs[f'{ROI}_R'])
            else:
                raise ValueError(f'ROI={ROI} not found')
        vecs = np.concatenate(vecs_l, axis=1)
        ROI2vecs_new[network] = vecs
    return ROI2vecs_new


def get_ROI_vecs_wrap(sn, atlas, fp0, df_sn, fp1=None, networks=None,
                      org_by_region=True, cross_region=False,
                      conn=None, combine_regions=False):
    assert not (combine_regions and org_by_region)
    ROI2vecs0 = get_ROI_vecs(sn, atlas, fp0, df_sn, nan_thresh=1.01,
                             drop_nan_voxels=False,
                             org_by_region=org_by_region,
                             easy_override=False,
                             combine_regions=combine_regions)
    if fp1 is not None:
        ROI2vecs1 = get_ROI_vecs(sn, atlas, fp1, df_sn, nan_thresh=1.01,
                                 drop_nan_voxels=False,
                                 org_by_region=org_by_region,
                                 easy_override=False,
                                 combine_regions=combine_regions)
    else:
        ROI2vecs1 = None

    if networks:
        if cross_region:
            ROI2vecs0 = calculate_cross_region_vecs(ROI2vecs0, networks,
                                                    conn=conn)
        else:
            ROI2vecs0 = cluster_regions(ROI2vecs0, networks)

        if fp1 is not None:
            if cross_region:
                ROI2vecs1 = calculate_cross_region_vecs(ROI2vecs1, networks,
                                                        conn=conn)
            else:
                ROI2vecs1 = cluster_regions(ROI2vecs1, networks)

    if fp1 is not None:
        return ROI2vecs0, ROI2vecs1
    else:
        return ROI2vecs0


def get_trial_x_trial(vecs, vecs1=None, trial_similarity='corr'):
    # Fixed bug (Nov 21, where this was stdizing for euc & seuc)
    vecs0 = vecs[None, :, :]
    if vecs1 is not None:
        has_vecs1 = True
        vecs1 = vecs1[:, None, :]
    else:
        has_vecs1 = False
        vecs1 = vecs[:, None, :]
    if trial_similarity == 'corr':
        vecs0 = stdize(vecs0, axis=2, nans=True)
        vecs1 = stdize(vecs1, axis=2, nans=True)
        RSM_fMRI = np.nanmean(vecs0 * vecs1, axis=-1)  # Pearson
    elif trial_similarity == 'euc':
        RSM_fMRI = np.nanmean(-abs(vecs0 - vecs1), axis=-1)  # Euclidean
        RSM_fMRI = prune_RSM_outliers(RSM_fMRI, z=3)

    elif trial_similarity == 'spear':
        # t = time()
        vecs0_r = stats.rankdata(vecs0, method='average', axis=-1)
        vecs1_r = stats.rankdata(vecs1, method='average', axis=-1)
        vecs0_r = stdize(vecs0_r, axis=2, nans=True)
        vecs1_r = stdize(vecs1_r, axis=2, nans=True)

        RSM_fMRI = np.nanmean(vecs0_r * vecs1_r, axis=-1)  # Spearman
        # print(f'Time needed for spearman trial x trial: {time() - t:.2f}s')
        # plt.imshow(RSM_fMRI)
        # plt.title('Fast')
        # plt.colorbar()
        # plt.show()
        # t = time()
        #
        # RSM_fMRI = np.nanmean(-abs(vecs0 - vecs1), axis=-1)  # Spearman
        # print(f'Time needed for spearman trial x trial: {time() - t:.2f}s')
        # # prune_RSM_outliers(RSM_fMRI, z=3.5)
        #
        # plt.imshow(RSM_fMRI)
        # plt.title('euc')
        # plt.colorbar()
        # plt.show()
        #
        # RSM_fMRI = np.nanmean(-abs(vecs0_r - vecs1_r), axis=-1)  # Spearman
        # print(f'Time needed for spearman trial x trial: {time() - t:.2f}s')
        # plt.imshow(RSM_fMRI)
        # plt.title('ranked euc')
        # plt.colorbar()
        # plt.show()
        # t = time()
        #
        #
        # RSM_fMRI = scipy_dist(np.squeeze(vecs0), np.squeeze(vecs1),
        #                       metric='seuclidean')
        # # prune_RSM_outliers(RSM_fMRI, z=3.5)
        # plt.imshow(RSM_fMRI)
        # plt.title('std euc')
        # plt.colorbar()
        # plt.show()
        # t = time()
        #
        #
        #
        # print(f'{has_vecs1=}')
        #
        # vecs0, vecs1 = np.squeeze(vecs0), np.squeeze(vecs1)
        # RSM_fMRI = np.zeros((vecs0.shape[0], vecs1.shape[0]))
        # for i in range(vecs0.shape[0]):
        #     for j in range(vecs1.shape[0]):
        #         if i >= j and not has_vecs1:
        #             continue
        #         r, p = stats.spearmanr(vecs0[i], vecs1[j])
        #         z = np.arctanh(r)
        #         RSM_fMRI[j, i] = z
        #         if not has_vecs1:
        #             RSM_fMRI[j, i] = z
        # print(f'Time needed for spearman trial x trial: {time() - t:.2f}s')
        #
        # plt.imshow(RSM_fMRI)
        # plt.colorbar()
        # plt.title('Slow spearman')
        # plt.show()
        # quit()
        # RSM_fMRI[np.diag_indices_from(RSM_fMRI)] = np.nan
    elif trial_similarity == 'seuclidean':
        RSM_fMRI = scipy_dist(np.squeeze(vecs0), np.squeeze(vecs1),
                              metric='seuclidean')
        RSM_fMRI = prune_RSM_outliers(RSM_fMRI, z=3)
    else:
        raise ValueError(f'{trial_similarity=} not supported')
    return RSM_fMRI


def get_BNA_ROIs():
    ROIs = ['1 SFG_L_7_1', '2 SFG_R_7_1', '3 SFG_L_7_2', '4 SFG_R_7_2', '5 SFG_L_7_3', '6 SFG_R_7_3', '7 SFG_L_7_4', '8 SFG_R_7_4', '9 SFG_L_7_5', '10 SFG_R_7_5', '11 SFG_L_7_6', '12 SFG_R_7_6', '13 SFG_L_7_7', '14 SFG_R_7_7', '15 MFG_L_7_1', '16 MFG_R_7_1', '17 MFG_L_7_2', '18 MFG_R_7_2', '19 MFG_L_7_3', '20 MFG_R_7_3', '21 MFG_L_7_4', '22 MFG_R_7_4', '23 MFG_L_7_5', '24 MFG_R_7_5', '25 MFG_L_7_6', '26 MFG_R_7_6', '27 MFG_L_7_7', '28 MFG_R_7_7', '29 IFG_L_6_1', '30 IFG_R_6_1', '31 IFG_L_6_2', '32 IFG_R_6_2', '33 IFG_L_6_3', '34 IFG_R_6_3', '35 IFG_L_6_4', '36 IFG_R_6_4', '37 IFG_L_6_5', '38 IFG_R_6_5', '39 IFG_L_6_6', '40 IFG_R_6_6', '41 OrG_L_6_1', '42 OrG_R_6_1', '43 OrG_L_6_2', '44 OrG_R_6_2', '45 OrG_L_6_3', '46 OrG_R_6_3', '47 OrG_L_6_4', '48 OrG_R_6_4', '49 OrG_L_6_5', '50 OrG_R_6_5', '51 OrG_L_6_6', '52 OrG_R_6_6', '53 PrG_L_6_1', '54 PrG_R_6_1', '55 PrG_L_6_2', '56 PrG_R_6_2', '57 PrG_L_6_3', '58 PrG_R_6_3', '59 PrG_L_6_4', '60 PrG_R_6_4', '61 PrG_L_6_5', '62 PrG_R_6_5', '63 PrG_L_6_6', '64 PrG_R_6_6', '65 PCL_L_2_1', '66 PCL_R_2_1', '67 PCL_L_2_2', '68 PCL_R_2_2', '69 ATL_L_6_1', '70 ATL_R_6_1', '71 STG_L_6_2', '72 STG_R_6_2', '73 STG_L_6_3', '74 STG_R_6_3', '75 STG_L_6_4', '76 STG_R_6_4', '77 ATL_L_6_5', '78 ATL_R_6_5', '79 ATL_L_6_6', '80 ATL_R_6_6', '81 MTG_L_4_1', '82 MTG_R_4_1', '83 ATL_L_4_2', '84 ATL_R_4_2', '85 MTG_L_4_3', '86 MTG_R_4_3', '87 MTG_L_4_4', '88 MTG_R_4_4', '89 ITG_L_7_1', '90 ITG_R_7_1', '91 ITG_L_7_2', '92 ITG_R_7_2', '93 ATL_L_7_3', '94 ATL_R_7_3', '95 ITG_L_7_4', '96 ITG_R_7_4', '97 ITG_L_7_5', '98 ITG_R_7_5', '99 ITG_L_7_6', '100 ITG_R_7_6', '101 ITG_L_7_7', '102 ITG_R_7_7', '103 FuG_L_3_1', '104 FuG_R_3_1', '105 FuG_L_3_2', '106 FuG_R_3_2', '107 FuG_L_3_3', '108 FuG_R_3_3', '109 PhG_L_6_1', '110 PhG_R_6_1', '111 PhG_L_6_2', '112 PhG_R_6_2', '113 PhG_L_6_3', '114 PhG_R_6_3', '115 PhG_L_6_4', '116 PhG_R_6_4', '117 PhG_L_6_5', '118 PhG_R_6_5', '119 PhG_L_6_6', '120 PhG_R_6_6', '121 pSTS_L_2_1', '122 pSTS_R_2_1', '123 pSTS_L_2_2', '124 pSTS_R_2_2', '125 SPL_L_5_1', '126 SPL_R_5_1', '127 SPL_L_5_2', '128 SPL_R_5_2', '129 SPL_L_5_3', '130 SPL_R_5_3', '131 SPL_L_5_4', '132 SPL_R_5_4', '133 SPL_L_5_5', '134 SPL_R_5_5', '135 IPL_L_6_1', '136 IPL_R_6_1', '137 IPL_L_6_2', '138 IPL_R_6_2', '139 IPL_L_6_3', '140 IPL_R_6_3', '141 IPL_L_6_4', '142 IPL_R_6_4', '143 IPL_L_6_5', '144 IPL_R_6_5', '145 IPL_L_6_6', '146 IPL_R_6_6', '147 Pcun_L_4_1', '148 Pcun_R_4_1', '149 Pcun_L_4_2', '150 Pcun_R_4_2', '151 Pcun_L_4_3', '152 Pcun_R_4_3', '153 Pcun_L_4_4', '154 Pcun_R_4_4', '155 PoG_L_4_1', '156 PoG_R_4_1', '157 PoG_L_4_2', '158 PoG_R_4_2', '159 PoG_L_4_3', '160 PoG_R_4_3', '161 PoG_L_4_4', '162 PoG_R_4_4', '163 INS_L_6_1', '164 INS_R_6_1', '165 INS_L_6_2', '166 INS_R_6_2', '167 INS_L_6_3', '168 INS_R_6_3', '169 INS_L_6_4', '170 INS_R_6_4', '171 INS_L_6_5', '172 INS_R_6_5', '173 INS_L_6_6', '174 INS_R_6_6', '175 CG_L_7_1', '176 CG_R_7_1', '177 CG_L_7_2', '178 CG_R_7_2', '179 CG_L_7_3', '180 CG_R_7_3', '181 CG_L_7_4', '182 CG_R_7_4', '183 CG_L_7_5', '184 CG_R_7_5', '185 CG_L_7_6', '186 CG_R_7_6', '187 CG_L_7_7', '188 CG_R_7_7', '189 EVC_L_5_1', '190 EVC_R_5_1', '191 EVC_L_5_2', '192 EVC_R_5_2', '193 EVC_L_5_3', '194 EVC_R_5_3', '195 EVC_L_5_4', '196 EVC_R_5_4', '197 EVC_L_5_5', '198 EVC_R_5_5', '199 LOC_L_4_1', '200 LOC_R_4_1', '201 LOC_L_4_2', '202 LOC_R_4_2', '203 LOC_L_4_3', '204 LOC_R_4_3', '205 LOC_L_4_4', '206 LOC_R_4_4', '207 sOcG_L_2_1', '208 sOcG_R_2_1', '209 sOcG_L_2_2', '210 sOcG_R_2_2', '211 Amyg_L_2_1', '212 Amyg_R_2_1', '213 Amyg_L_2_2', '214 Amyg_R_2_2', '215 Hipp_L_2_1', '216 Hipp_R_2_1', '217 Hipp_L_2_2', '218 Hipp_R_2_2', '219 Str_L_6_1', '220 Str_R_6_1', '221 Str_L_6_2', '222 Str_R_6_2', '223 Str_L_6_3', '224 Str_R_6_3', '225 Str_L_6_4', '226 Str_R_6_4', '227 Str_L_6_5', '228 Str_R_6_5', '229 Str_L_6_6', '230 Str_R_6_6', '231 Tha_L_8_1', '232 Tha_R_8_1', '233 Tha_L_8_2', '234 Tha_R_8_2', '235 Tha_L_8_3', '236 Tha_R_8_3', '237 Tha_L_8_4', '238 Tha_R_8_4', '239 Tha_L_8_5', '240 Tha_R_8_5', '241 Tha_L_8_6', '242 Tha_R_8_6', '243 Tha_L_8_7', '244 Tha_R_8_7', '245 Tha_L_8_8', '246 Tha_R_8_8']
    return ROIs
