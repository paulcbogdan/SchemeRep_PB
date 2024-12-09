import utils
from Utils.atlas_funcs import get_atlas
from connRSA_finalizing.conn_Fig6 import run_IC_analysis
from scipy import spatial, stats
import numpy as np
from nilearn import image, plotting

def map_plot_ROI(nearest=5):
    corr = utils.pickle_wrap(run_IC_analysis, kwargs={'ERS': True,
                                                      'regress_FC': False,
                                                      'get_M': True},)
    print(corr.shape)

    atlas = get_atlas()


    coords = atlas['coords']

    euc_mtx = spatial.distance.cdist(coords, coords)
    euc_mtx = np.array(euc_mtx, dtype=np.int8)

    n_roi = len(coords)
    img_data = np.zeros(atlas['maps'].shape)
    for roi in range(n_roi):
        nearest_idxs = np.argsort(euc_mtx[roi])[1:nearest+1]
        corr = corr[:] - np.nanmean(corr, axis=(1, 2))[:, None, None]
        M_RSM = corr[:, roi, nearest_idxs]
        # M_RSM = corr[:, roi, nearest_idxs]

        M_RSM = np.nanmean(M_RSM, axis=-1)
        # print(M_RSM.shape)
        # quit()
        # M
        t = stats.ttest_1samp(M_RSM, 0)
        # M_
        # img_data[atlas['maps'].get_fdata() == roi + 1] = np.nanmean(M_RSM)
        img_data[atlas['maps'].get_fdata() == roi + 1] = t.statistic
        # print(np.sum(atlas['maps'].get_fdata() == roi))
        # quit()

    # vmin = np.nanquantile(img_data, 0.01)
    # vmax = np.nanquantile(img_data, 0.99)
    #
    img = image.new_img_like(atlas['maps'], img_data)

    # print(atlas['maps'].shape)
    # print(img.get_fdata()[40, 50, :])
    # quit()
    view = plotting.view_img(img, threshold=0.01, symmetric_cmap=False,
                             resampling_interpolation='nearest',
                             # vmin=vmin, vmax=vmax
                             )
    view.open_in_browser()
    # euc_mtx = utils.get_euc_mtx(coords)

    # corr = run_IC_analysis(ERS=True, regress_FC=True, get_M=False)

if __name__ == '__main__':
    map_plot_ROI()