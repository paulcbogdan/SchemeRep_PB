import pickle
import numpy as np

from ROIs import get_atlas
from plotting import plot_connectivity
from utils import get_cache_RSA_fp, stdize


def get_conn_mat(age=1, early=True, semantic=False, cin=None,
                 bilateral=False, combine_regions=False, vec_prod=False,
                 org_by_region=False, rxr=False):
    fp = get_cache_RSA_fp(cin=cin, age=age, semantic=semantic, early=early,
                          bilateral=bilateral, combine_regions=combine_regions,
                          vec_prod=vec_prod, org_by_region=org_by_region,
                          rxr=rxr)

    with open(fp, 'rb') as file:
        d = pickle.load(file)

    atlas = get_atlas(combine_regions=combine_regions or org_by_region,
                      bilateral=bilateral or org_by_region)

    sn_time_series = np.full((33, 246, 114), np.nan)
    for i, roi in enumerate(d['activity']):
        # sn_time_series[:, i, :] = d['activity'][roi]
        sn_time_series[:, i, :] = d['IRAFs_ROI']['scn'][roi]

    sn_mats = np.full((33, 246, 246), np.nan)
    for sn in range(sn_time_series.shape[0]):
        for j in range(sn_time_series.shape[2]):
            sn_time_series[sn, :, j] = stdize(sn_time_series[sn, :, j])
        sn_mats[sn, :, :] = np.corrcoef(sn_time_series[sn, :, :])
        sn_mats[sn, :, :] = np.arctanh(sn_mats[sn, :, :])

    ar = np.nanmean(sn_mats, axis=0)

    ar[np.diag_indices_from(ar)] = np.nan

    plot_connectivity(ar, atlas['ticks'], atlas['tick_labels'],
                      atlas['tick_lows'],
                      title='', no_avg=True)


if __name__  == '__main__':
    get_conn_mat()

