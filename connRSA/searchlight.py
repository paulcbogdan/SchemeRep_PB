from pathlib import Path
from time import time

from numba import jit, njit, prange, set_num_threads

import utils
from connRSA.single_trial_conn import prep_vecs, prep_fps
from org_sns import get_sns
from organize_bhv import get_trial_info
import numpy as np
import matplotlib.pyplot as plt

from stim import get_stim_RDM


@njit(parallel=True, fastmath=True, nopython=True)
def jit_searchlight(img, nan_mask, radius, RSM, skip_step=1):
    X_len = img.shape[0]
    Y_len = img.shape[1]
    Z_len = img.shape[2]
    num_do = 0
    out = np.full((X_len, Y_len, Z_len), np.nan, dtype=np.float64)
    tril_mask = np.tril_indices_from(RSM, k=-1)

    stim_flat = np.full(6441, np.nan)
    for cnt in range(6441): # can't fancy index with numba???
        stim_flat[cnt] = RSM[tril_mask[0][cnt], tril_mask[1][cnt]]

    size = (2*radius+1)**3

    for x in prange(radius, X_len-radius):
    # for x in range(60, 65):
        for y in range(radius, Y_len-radius, skip_step):
            for z in range(radius, Z_len-radius, skip_step):
                # nan_sq = nans[x-radius:x+radius+1,
                #               y-radius:y+radius+1,
                #               z-radius:z+radius+1]
                # nan_sq = nan_mask[x-1:x+2,
                #                   y-1:y+2,
                #                   z-1:z+2]
                # nan_sq = nan_mask[x - 1:x + 2,
                #          y - 1:y + 2,
                #          z - 1:z + 2]
                # if np.any(nan_sq):
                #     continue
                if nan_mask[x, y, z]:
                    continue

                square_flat = np.full((114, size), np.nan)
                cnt = 0
                for di in range(-radius, radius+1):
                    for dj in range(-radius, radius+1):
                        for dk in range(-radius, radius+1):
                            square_flat[:, cnt] = img[x+di, y+dj, z+dk, :]
                            cnt += 1

                # nans_flat = np.isnan(square_flat[0, :])

                # TODO: I could memoize this??
                nans_flat = np.full(size, False)
                for i in range(size): # NaNs dont work or something??
                    nans_flat[i] = np.any(square_flat[:, i] > 10000)
                    # nans_flat[i] = np.any(np.isnan(square_flat[:, i]))
                    # print(i, nans_flat[i])
                    # print(square_flat[:, i])
                    # print(np.isnan(square_flat[:, i]))
                # nans_flat = np.any(np.isnan(square_flat), axis=0)
                square_flat = square_flat[:, ~nans_flat] # split spheres are fine
                # print(square_flat)
                fMRI_RSM = np.corrcoef(square_flat)
                fMRI_flat = np.full(6441, np.nan)
                for cnt in range(6441):
                    fMRI_flat[cnt] = fMRI_RSM[tril_mask[0][cnt], tril_mask[1][cnt]]
                out[x, y, z] = np.corrcoef(stim_flat, fMRI_flat)[0, 1]
                # print(f'{x}, {y}, {z}: {out[x, y, z]}')
                # print(x, ',', y, ',', z, ':', out[x, y, z])
                num_do += 1
                # return
        #         break
        #     if num_do:
        #         break
        # if num_do:
        #     break
    return out



# fp_fMRI_col='bl7_fMRI',
def test_searchlight(semantic=True, radius=2, skip_step=1):
    age2sn = get_sns('all', sh=False)
    sns = age2sn[1] + age2sn[2]
    d_vecs = prep_vecs(True, semantic)
    fps = prep_fps('7')
    for fp_fMRI_col in fps:
        for sn in sns:
            df_sn = get_trial_info(sn)
            img, good_idxs = utils.load_ni_w_nan_fps(df_sn[fp_fMRI_col])
            flat = img.flatten()

            nans = np.all(np.isnan(img), axis=3)
            # mask = nans
            # plt.imshow(nans[60, :, :])
            # plt.show()

            img[np.isnan(img)] = 10001

            RSM_stim = get_stim_RDM(df_sn, d_vecs, obj_only=True, dist='corr')

            dir_save = f'cache/searchlight'
            Path(dir_save).mkdir(parents=True, exist_ok=True)
            fp_save = (f'{dir_save}/{sn}_{fp_fMRI_col}_{radius}_{semantic}_'
                       f'{skip_step}.pkl')

            t_st = time()
            kw = {'img': img, 'nan_mask': nans, 'radius': radius,
                  'RSM': RSM_stim, 'skip_step': skip_step}
            # searched = utils.pickle_wrap(jit_searchlight, fp_save, kw,
            #                              verbose=1, easy_override=True)

            searched = utils.pickle_wrap(lambda: jit_searchlight(**kw), fp_save,
                                         verbose=-1, easy_override=True)

            plt.title(f'{fp_save}')
            plt.imshow(searched[60, :, :])
            plt.show()
            # quit()
            t_end = time()
            # print(f'{img.shape=}')
            t_needed = t_end - t_st
            print(f'{t_needed=:.2f}, {kw=}')



            # quit()

# from numba import njit, prange, set_num_threads
set_num_threads(4)

if __name__ == '__main__':
    test_searchlight()


