from collections import defaultdict
from time import time

import numpy as np
from tqdm import tqdm

from atlas_utils import get_atlas
from fMRI_proc import get_ROI_vecs
from org_sns import get_sns, get_shenyang_subjects
from organize_bhv import get_trial_info
import scipy.stats as stats

from utils import stdize


def correlate_activity(fp0='obj3_fMRI', fp1='cmb3_fMRI'):
    atlas = get_atlas(combine_regions=False, combine_bilateral=False,
                      shenyang=True)
    coords = atlas['coords']
    sh_sns = get_shenyang_subjects()
    age2sn = get_sns(fp0, sh=False)

    for i, age in enumerate([1, 2]):
        sns = age2sn[age]
        rs_all = []
        vals0 = defaultdict(list)
        vals1 = defaultdict(list)
        difs = defaultdict(list)
        for sn in sns:
            if sn == '111':
                continue
            print(f'Prepping univariate: {sn=}')
            df_sn = get_trial_info(sn, easy_override=False)

            ROI2vecs0 = get_ROI_vecs(sn, atlas, fp0, df_sn, nan_thresh=1.01,
                                     drop_nan_voxels=False,
                                     org_by_region=False,
                                     easy_override=False,
                                     combine_regions=False)

            ROI2vecs1 = get_ROI_vecs(sn, atlas, fp1, df_sn, nan_thresh=1.01,
                                     drop_nan_voxels=False,
                                     org_by_region=False,
                                     easy_override=False,
                                     combine_regions=False)

            rs = []
            # activity0 = [act0 for act0 in ROI2vecs0.values()]
            # activity0 = np.concatenate(activity0, axis=1)
            # activity0 = activity0.T
            #
            # activity1 = [act1 for act1 in ROI2vecs1.values()]
            # activity1 = np.concatenate(activity1, axis=1)
            # activity1 = activity1.T
            # st = time()
            for ROI in tqdm(ROI2vecs0):
                # if 'pSTS_L_2_1' not in ROI:
                #     continue
                activity0 = ROI2vecs0[ROI].T
                activity0 = np.nanmean(activity0, axis=0)[None, :]
                vals0[ROI].append(np.nanmean(activity0))
                # print(np.nanmean(activity0))
                # quit()
                # print(f'{vals0[ROI]=}')
                # print(ROI)
                # quit()
                # print(activity0.shape)
                # quit()
                #
                activity1 = ROI2vecs1[ROI].T
                activity1 = np.nanmean(activity1, axis=0)[None, :]
                vals1[ROI].append(np.nanmean(activity1))

                # activity0 = ROI2vecs0[ROI].T
                # activity0 = np.nanmean(activity0, axis=1)[None, :]
                # #
                # activity1 = ROI2vecs1[ROI].T
                # activity1 = np.nanmean(activity1, axis=1)[None, :]

                activity0 = stats.rankdata(activity0, axis=1,
                                           nan_policy='omit')
                activity1 = stats.rankdata(activity1, axis=1,
                                           nan_policy='omit')
                activity0 = stdize(activity0, axis=1, nans=True)
                activity1 = stdize(activity1, axis=1, nans=True)
                rs_ROI = np.nanmean(activity0 * activity1, axis=1)
                rs.append(rs_ROI)
            rs = np.concatenate(rs)
            M_rs = np.nanmedian(rs)
            M_rsqs = np.nanmedian(rs ** 2)
            print(f'{fp0=}, {fp1=} | {sn}, {M_rs=:.3f}, {M_rsqs=:.3f}')
            rs_all.append(M_rsqs)
            key = '121 pSTS_L_2_1'
            # print(vals0[key])
            t_obj, _ = stats.ttest_1samp(vals0[key], 0)
            t_scn, _ = stats.ttest_1samp(vals1[key], 0)
            t_dif, _ = stats.ttest_rel(vals0[key], vals1[key])
            print(f'{t_obj=:.3f}, {t_scn=:.3f}, {t_dif=:.3f}')
            print()



        rs_all = np.array(rs_all)
        print(f'{fp0=}, {fp1=}, {age=}, {np.nanmean(rs_all)=:.4f}')



if __name__ == '__main__':
    correlate_activity(fp0='obj7_fMRI', fp1='scn7_fMRI')

    # correlate_activity(fp0='scn3_fMRI')









