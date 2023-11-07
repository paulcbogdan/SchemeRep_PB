import numpy as np

from analyze_ROIs import prune_bad_sns
from plot_gen import plot_connectivity
from utils import make_title_str, get_RSA_fn, stdize
from atlas_utils import get_atlas
import matplotlib.pyplot as plt
import pickle
import pandas as pd

import statsmodels.formula.api as smf


def replace_w_nan_if_needed(vals):
    clean = []
    for x in vals:
        if x.shape == vals[0].shape:
            clean.append(x)
        else:
            shape_nan = (vals[0].shape[0] - x.shape[0], vals[0].shape[1])
            fill_nan = np.full(shape_nan, np.nan)
            x = np.concatenate([x, fill_nan])
            clean.append(x)
            # print(f'{x.shape=}')
            # quit()
            # clean.append(np.full(vals[0].shape, np.nan))
    return np.array(clean)

def random_interaction_stuff(d, d2):
    for roi in ['SFG_L', 'SFG_R', 'MFG_L', 'MFG_R', 'IFG_L', 'IFG_R']:
        DV = np.array(d['IRAFs_ROI']['dif_abs'][roi])
        DV = stdize(DV, nans=True)
        DV = np.reshape(DV, -1)
        IV0 = np.array(d['IRAFs_ROI']['obj'][roi])
        IV0 = stdize(IV0, nans=True)
        IV0 = np.reshape(IV0, -1)
        IV1 = np.array(d['IRAFs_ROI']['scn'][roi])
        IV1 = stdize(IV1, nans=True)
        IV1 = np.reshape(IV1, -1)
        sns = np.array([[sn]*114 for sn in d['sns']])
        sns = np.reshape(sns, -1)
        df = pd.DataFrame({'DV': DV, 'IV0': IV0, 'IV1': IV1, 'sns': sns})
        res = smf.ols('DV ~ IV0*IV1', data=df).fit()
        print(res.summary())
        print(f'{roi=}')
        print('-'*100)
    quit()

def test_IRAF_x_activity(age=1, early=True, semantic=False, cin=None,
                 bilateral=False, combine_regions=True, vec_prod=False,
                 org_by_region=False, PCA_obj=True):
    fn = get_RSA_fn(inc=cin, age=age, semantic=semantic, DNN_layer=2,
                    fp_fMRI_col='obj_fMRI',
                    bilateral=bilateral, combine_regions=combine_regions,
                    vec_prod=vec_prod, org_by_region=org_by_region,
                    PCA_obj=PCA_obj)
    fp = fr'cache/RSA/{fn}.pkl'
    with open(fp, 'rb') as file:
        d = pickle.load(file)

    fn2 = get_RSA_fn(inc=cin, age=age, semantic=semantic, DNN_layer=2,
                     fp_fMRI_col='obj_fMRI',
                     bilateral=bilateral, combine_regions=combine_regions,
                     vec_prod=vec_prod, org_by_region=org_by_region,
                     PCA_obj=PCA_obj)
    fp2 = fr'cache/RSA/{fn2}.pkl'
    with open(fp2, 'rb') as file:
        d2 = pickle.load(file)
    # d = prune_bad_sns(d, drop_ret=True)

    atlas = get_atlas(combine_regions=combine_regions or org_by_region,
                      bilateral=bilateral or org_by_region)
    # Can correlated dif_abs IRAF x obj IRAF, both for obj_fMRI?
    key = 'obj'
    left = [np.array(d['IRAFs_ROI']['dif_abs_'][roi0]) for roi0 in atlas['ROIs']]
    # bottom = [np.array(d2['IRAFs_ROI']['obj'][roi0]) for roi0 in atlas['ROIs']]
    bottom = [np.array(d['activity'][roi0]) for roi0 in atlas['ROIs']]
    # bottom = [np.array(d['IRAFs_ROI']['obj'][roi0]) for roi0 in atlas['ROIs']]

    #
    # left = [np.array(d['IRAFs_ROI']['obj'][roi0]) for roi0 in atlas['ROIs']]
    # bottom = [np.array(d2['IRAFs_ROI']['obj'][roi0]) for roi0 in atlas['ROIs']]
    #
    # left = [np.array(d['IRAFs_ROI']['dif_abs_'][roi0]) for roi0 in atlas['ROIs']]
    # bottom = [np.array(d2['IRAFs_ROI']['obj'][roi0]) for roi0 in atlas['ROIs']]




    # IRAFs = [np.array(d['activity'][roi1]) for roi1 in atlas['ROIs']]

    # activity = [np.array(d['IRAFs_ROI']['dif_abs'][roi1]) for roi1 in atlas['ROIs']]
    # activity = [np.array(d2['activity'][roi1]) for roi1 in atlas['ROIs']]

    # activity = [np.array(d2['IRAFs_ROI']['obj'][roi0]) for roi0 in atlas['ROIs']]

    # activity = [np.array(d2['IRAFs_ROI']['scn'][roi0]) for roi0 in atlas['ROIs']]
    # random_interaction_stuff(d, d2)
    # quit()
    # activity = [np.array(d2['IRAFs_ROI']['obj'][roi0]) for roi0 in atlas['ROIs']]
    # activity = [np.array(d2['IRAFs_ROI']['scn'][roi0]) for roi0 in atlas['ROIs']]

    # IRAFs = np.array(IRAFs)
    # for i in range(31):
    #     for j in range(5, 20):
    #         IRAFs[j, i, :] = np.arange(114)
    #
    #
    #
    #
    # activity = np.array(activity)
    # for i in range(31):
    #     activity[5:10, i, :] = np.arange(114) + np.random.random(114) * .0001


    bottom = replace_w_nan_if_needed(bottom)

    r_Ms, r_SDs, t, _ = bulk_correlate(left, bottom, nans=True)
    # print(r_Ms.shape)
    # quit()
    # t[np.diag_indices_from(t)] = np.nan
    title = make_title_str('', key, age, early, semantic, cin)
    plot_connectivity(t, atlas['ticks'], atlas['tick_labels'],
                      atlas['tick_lows'],
                      title=title, no_avg=True,
                      cbar_label='t-value', vmin=-4, vmax=4)
    # vmin=-3

def bulk_correlate(vals0, vals1, nans=False):
    # Weird, takes in ar.shape = (n_ROIs, n_subjs, n_timepoints)

    vals0 = np.expand_dims(vals0, axis=1)
    vals1 = np.expand_dims(vals1, axis=0)

    m = np.nanmean if nans else np.mean
    s = np.nanstd if nans else np.std
    vals0_M = m(vals0, axis=-1)

    vals0_SD = s(vals0, axis=-1)
    vals0_ = (vals0 - vals0_M[:, :, :, None]) / vals0_SD[:, :, :, None]
    vals1_M = m(vals1, axis=-1)
    vals1_SD = s(vals1, axis=-1)
    vals1_ = (vals1 - vals1_M[:, :, :, None]) / vals1_SD[:, :, :, None]

    rs = vals0_ * vals1_
    rs = m(rs, axis=-1)
    # rs = np.arctanh(rs)
    r_Ms = np.nanmean(rs, axis=-1)
    # r_Ms[np.diag_indices_from(r_Ms)] = np.nan
    r_SDs = np.nanstd(rs, axis=-1)
    # r_SDs[np.diag_indices_from(r_Ms)] = np.nan
    # print(rs.shape[-1])
    # print(r_Ms.shape)
    # quit()
    # print(r_Ms.shape)
    # print(rs[5, 48])
    # quit()

    n_nans = np.sum(~np.isnan(rs), axis=-1)
    t = r_Ms / r_SDs * np.sqrt(n_nans)
    return r_Ms, r_SDs, t, rs
    # print(rs.shape)
    # quit()
    # pass

def test_triple_z(age=1, early=True, semantic=False,
                         combine_regions=True, bilateral=False):
    fp = get_RSA_fn(inc=None, age=age, semantic=semantic, early=early,
                    combine_regions=combine_regions, bilateral=bilateral)
    with open(fp, 'rb') as file:
        d = pickle.load(file)
    # ROIs, ROI_nums, ticks, tick_labels, tick_lows, n_ROIs = add_ROI_info()
    atlas = get_atlas(combine_regions=combine_regions, bilateral=bilateral)
    data = d['triple_z']['obj']
    data = np.array(data)
    M = np.nanmean(data, axis=0)
    SD = np.nanstd(data, axis=0)
    t = M / SD * np.sqrt(len(data))
    plot_connectivity(t, atlas['ticks'], atlas['tick_labels'],
                      atlas['tick_lows'],
                      no_avg=True,
                      # title='YA, 1st-layer DNN RSA for objects. '
                      #       'triple-correlation',
                      title='YA, 1st-layer DNN object RSA, '
                            'voxel x voxel connectivity ',
                      cbar_label='t-value')

def permutation_testing_threshold():
    # TODO
    # Compare basic permutation testing (permute 1st level results) vs.
    #   detailed permutation testing (permute underlying data)
    # Mixed (efficient)
    # * Make 10-20 shuffled datasets per participant
    # * Get first-level results
    # * Sample from first-level results
    pass

def test_rxr(age=1, early=True, semantic=False, cin=None,
                 bilateral=False, combine_regions=True, vec_prod=False,
                 org_by_region=False, rxr=False):
    fp = get_RSA_fn(inc=cin, age=age, semantic=semantic, DNN_layer=2,
                    fp_fMRI_col='obj_fMRI',
                    bilateral=bilateral, combine_regions=combine_regions,
                    vec_prod=vec_prod, org_by_region=org_by_region,
                    )
    with open(fp, 'rb') as file:
        d = pickle.load(file)
    # ROIs, ROI_nums, ticks, tick_labels, tick_lows, n_ROIs = add_ROI_info()
    atlas = get_atlas(combine_regions=combine_regions or org_by_region,
                      bilateral=bilateral or org_by_region)
    data = d['triple_z']['obj']
    data = np.array(data)
    # for val in data[:, 41, 45]:
    #     print(val > 0)

    # print(data[:, 41, 45])
    # quit()

    M = np.nanmean(data, axis=0)
    SD = np.nanstd(data, axis=0)
    # t = M
    t = M / SD * np.sqrt(len(data))


    plot_connectivity(t, atlas['ticks'], atlas['tick_labels'],
                      atlas['tick_lows'],
                      no_avg=True,
                      # title='YA, 1st-layer DNN RSA for objects. '
                      #       'triple-correlation',
                      title='YA, 1st-layer DNN object RSA, '
                            'voxel x voxel connectivity ',
                      cbar_label='t-value')

    quit()


if __name__ == '__main__':
  test_IRAF_x_activity()
    # test_rxr()