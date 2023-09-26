import numpy as np

from plotting import plot_connectivity
from utils import make_title_str, get_cache_RSA_fp, stdize
from ROIs import get_atlas

import pickle





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


def test_IRAF_x_activity(age=1, early=True, semantic=False, cin=None,
                 bilateral=False, combine_regions=True, vec_prod=False,
                 org_by_region=False, rxr=False):
    fp = get_cache_RSA_fp(cin=cin, age=age, semantic=semantic, DNN_layer=2,
                          fp_fMRI_col='obj_fMRI',
                          bilateral=bilateral, combine_regions=combine_regions,
                          vec_prod=vec_prod, org_by_region=org_by_region,
                          rxr=rxr)
    with open(fp, 'rb') as file:
        d = pickle.load(file)

    fp2 = get_cache_RSA_fp(cin=cin, age=age, semantic=semantic, DNN_layer=2,
                          fp_fMRI_col='obj_fMRI',
                          bilateral=bilateral, combine_regions=combine_regions,
                          vec_prod=vec_prod, org_by_region=org_by_region,
                          rxr=rxr)
    with open(fp2, 'rb') as file:
        d2 = pickle.load(file)

    # print(np.array(d['bhv']['hit_bool']).shape)
    # print(d['IRAFs_ROI']['dif']['SFG_L'].shape)
    # test = np.full((33, 114), True)
    # d['bhv']['hit_bool'] = np.nan_to_num(d['bhv']['hit_bool'], True).astype(bool)
    # print(d['bhv']['hit_bool'])
    # test = d['bhv']['hit_bool']
    # d['IRAFs_ROI']['dif']['SFG_L'][~test] = np.nan
    # print( d['IRAFs_ROI']['dif']['SFG_L'])
    # quit()


    # ROIs, ROI_nums, ticks, tick_labels, tick_lows, n_ROIs = add_ROI_info()
    atlas = get_atlas(combine_regions=combine_regions or org_by_region,
                      bilateral=bilateral or org_by_region)
    key = 'dif_abs'
    # prune_to_only_hits(d, key, misses=False)
    IRAFs = [np.array(d['IRAFs_ROI'][key][roi0]) for roi0 in atlas['ROIs']]


    # IRAFs = [np.array(d['activity'][roi1]) for roi1 in atlas['ROIs']]
    # IRAFs = [np.array(d['activity'][roi0]) for roi0 in atlas['ROIs']]

    # IRAFs = replace_w_nan_if_needed(IRAFs)
    #
    # # for i in range(IRAFs.shape[1]):
    # #     np.random.shuffle(IRAFs[:, i, :])
    #     # if i % 2 == 0:
    #     # IRAFs[:, i, :] = np.hstack([IRAFs[:, i, ::2], IRAFs[:, i, ::2]])
    #     # else:
    #     # IRAFs[:, i, :] = np.hstack([IRAFs[:, i, 1::2], IRAFs[:, i, 1::2]])
    # # test = IRAFs[:, 2, :]
    # test = np.nanmedian(IRAFs, axis=1)
    # import matplotlib.pyplot as plt
    # plt.imshow(test)
    # plt.ylabel('ROI')
    # plt.xlabel('Item')
    # plt.title('Object, DNN Layer 2 (PCA)')
    # plt.colorbar()
    # plt.show()
    #
    # print(IRAFs.shape)
    #
    #
    # quit()
    # print(f'{IRAFs.shape=}')
    # quit()
    # IRAFs = np.array(list(filter(lambda x: x.shape[0] == 33, IRAFs)))
    activity = [np.array(d['IRAFs_ROI']['dif_abs'][roi1]) for roi1 in atlas['ROIs']]
    # activity = [np.array(d['activity'][roi1]) for roi1 in atlas['ROIs']]
    # activity = [np.array(d2['IRAFs_ROI']['dif'][roi0]) for roi0 in atlas['ROIs']]

    activity = replace_w_nan_if_needed(activity)

    # activity = activity - np.nanmean(activity[0], axis=0)

    # print(activity.shape)
    # print(IRAFs.shape)
    # quit()
    # activity = np.array(list(filter(lambda x: x.shape[0] == 33, activity)))
    # print(np.array(IRAFs).shape)
    r_Ms, r_SDs, t, _ = bulk_correlate(IRAFs, activity, nans=True)
    t[np.diag_indices_from(t)] = np.nan
    # print(t.shape)
    # quit()
    title = make_title_str('Activity x IRAF', key, age, early, semantic, cin)
    # title = 'YA. Activity x Difference-IRAF, 1st-layer DNN.'
    plot_connectivity(t, atlas['ticks'], atlas['tick_labels'],
                      atlas['tick_lows'],
                      title=title, no_avg=True,
                      cbar_label='t-value', vmin=-3)

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
    print('test', rs.shape)
    r_Ms = np.nanmean(rs, axis=-1)
    # r_Ms[np.diag_indices_from(r_Ms)] = np.nan
    r_SDs = np.nanstd(rs, axis=-1)
    # r_SDs[np.diag_indices_from(r_Ms)] = np.nan
    t = r_Ms / r_SDs * np.sqrt(rs.shape[-1]) # fix to account for different # nans per edge
    print(f'{t.shape=}')
    return r_Ms, r_SDs, t, rs
    # print(rs.shape)
    # quit()
    # pass

def test_triple_z(age=1, early=True, semantic=False,
                         combine_regions=True, bilateral=False):
    fp = get_cache_RSA_fp(cin=None, age=age, semantic=semantic, early=early,
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


def test_rxr(age=1, early=True, semantic=False, cin=None,
                 bilateral=False, combine_regions=False, vec_prod=False,
                 org_by_region=True, rxr=True):
    fp = get_cache_RSA_fp(cin=cin, age=age, semantic=semantic, early=early,
                          bilateral=bilateral, combine_regions=combine_regions,
                          vec_prod=vec_prod, org_by_region=org_by_region,
                          rxr=rxr)
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

    # t[41, 45] = 100
    # print(t[41, 43])
    # quit()
    # t = np.mean(data > 0, axis=0)
    plot_connectivity(t, atlas['ticks'], atlas['tick_labels'],
                      atlas['tick_lows'],
                      no_avg=True,
                      # title='YA, 1st-layer DNN RSA for objects. '
                      #       'triple-correlation',
                      title='YA, 1st-layer DNN object RSA, '
                            'voxel x voxel connectivity ',
                      cbar_label='t-value')


    # print(np.array(data).shape)
    quit()


if __name__ == '__main__':
    # test_triple_z()
    # CIN_compare()
    # analyze_ROIs()
    # test_rxr()
    test_IRAF_x_activity(early=True, semantic=False, cin=None,
                 bilateral=False, combine_regions=True, vec_prod=False,
                 org_by_region=False, rxr=False)



    # # Test connectivity within region between ROIs as nodes
    # analyze_ROIs(early=True, semantic=False, cin=None,
    #              bilateral=False, combine_regions=False, vec_prod=True,
    #              org_by_region=True, rxr=False)
