from time import time

from numba import jit, prange

import utils
from connRSA.single_trial_conn import prep_vecs
from org_sns import get_sns
from organize_bhv import get_trial_info
import numpy as np
import matplotlib.pyplot as plt

from stim import get_stim_RDM


@jit(parallel=True, fastmath=True, nopython=True)
def jit_searchlight(img, nan_mask, radius, RSM):
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

    # for x in prange(30, 35):
    # stim_flat = np.tril(RSM, k=-1)
    # print(stim_flat.shape)
    # quit()
    for x in prange(radius, X_len-radius):
        for y in range(radius, Y_len-radius):
            for z in range(radius, Z_len-radius):
                # nan_sq = nans[x-radius:x+radius+1,
                #               y-radius:y+radius+1,
                #               z-radius:z+radius+1]
                nan_sq = nan_mask[x - 1:x + 2,
                                  y-1:y+2,
                                  z-1:z+2]
                if np.any(nan_sq):
                    continue
                # if np.isnan()



                # square = img[x-radius:x+radius+1,
                #              y-radius:y+radius+1,
                #              z-radius:z+radius+1]
                # print(square.shape)
                # quit()
                square_flat = np.full((114, size), np.nan)

                cnt = 0
                for di in range(-radius, radius+1):
                    for dj in range(-radius, radius+1):
                        for dk in range(-radius, radius+1):
                            # print(f'{di}, {dj}, {dk}: {cnt}')
                            # cnt = ((di+radius)*size +
                            #        (dj+radius)*size +
                            #        (dk+radius))
                            # print(cnt)
                            square_flat[:, cnt] = img[x+di, y+dj, z+dk, :]
                            cnt += 1

                            # square_flat[:, (di+radius)*size + (dj+radius)*size + (dk+radius)] = img[x+di, y+dj, z+dk, :]
                # print(square_flat.shape)
                # quit()


                # square = np.transpose(square, (3, 0, 1, 2))
                # print(square.shape)

                # flat = np.reshape(square, [square.shape[0], -1])
                # flat = square.reshape(square.shape[0], -1)
                # flat = np.reshape(square, (square.shape[0], -1))

                # print(square.shape)
                # print(flat.shape)
                # quit()
                nans_flat = np.isnan(square_flat[0, :])
                square_flat = square_flat[:, ~nans_flat] # split spheres are fine
                # print(nans.shape)
                # plt.imshow(flat)
                # plt.show()
                # print(flat.shape)
                # quit()
                fMRI_RSM = np.corrcoef(square_flat)
                # fMRI_flat = fMRI_RSM[tril_mask]

                fMRI_flat = np.full(6441, np.nan)

                for cnt in range(6441):
                    fMRI_flat[cnt] = fMRI_RSM[tril_mask[0][cnt], tril_mask[1][cnt]]

                # out[x, y, z] = np.correlate(stim_flat, fMRI_flat)
                out[x, y, z] = np.corrcoef(stim_flat, fMRI_flat)[0, 1]
                # print(out[x, y, z])
                num_do += 1
                # print(f'{x}, {y}, {z}: {out[x, y, z]}')
    # print(f'{num_do=}')
    # quit()
    return out



def test_searchlight(fp_fMRI_col='bl7_fMRI', semantic=True):
    age2sn = get_sns('all', sh=False)
    sns = age2sn[1] + age2sn[2]
    d_vecs = prep_vecs(True, semantic)

    for sn in sns:
        df_sn = get_trial_info(sn)
        img, good_idxs = utils.load_ni_w_nan_fps(df_sn[fp_fMRI_col])
        flat = img.flatten()
        # plt.imshow(img[31, :, :, 20])
        # plt.show()
        # quit()
        nans = np.all(np.isnan(img), axis=3)
        # mask = nans

        RSM_stim = get_stim_RDM(df_sn, d_vecs, obj_only=True, dist='corr')

        t_st = time()
        searched = jit_searchlight(img, nans, 2, RSM_stim)
        t_end = time()
        # print(f'{img.shape=}')
        t_needed = t_end - t_st
        print(f'{t_needed=:.2f}')



        # quit()


if __name__ == '__main__':
    test_searchlight()


