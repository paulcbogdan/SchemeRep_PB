import numpy as np
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

from Utils.atlas_funcs import get_BNA_ROIs, get_atlas
from connRSA.DistRep_ROI_RSA import ROI2NETWORK
from connRSA.make_RSM_stim import get_sn_fp_stim_RSM
from fMRI_proc import within_run_to_nan
from networks.old.networks import prep_networks
from org_sns import get_sns
from organize_bhv import get_trial_info
import utils
from functools import cache
import matplotlib.pyplot as plt

from stim import get_semantic_vectors, get_DNN_vecs
import scipy.stats as stats
from time import time

# suppress RuntimeWarning: invalid value encountered in divide
np.seterr(divide='ignore', invalid='ignore')

def get_ROIs_from_region(region):
    regions = set(prep_networks(
        network_setting=ROI2NETWORK[region])[region])
    ROIs_match = [ROI for region in regions
                  for ROI in get_BNA_ROIs() if region in ROI]
    return ROIs_match
    # ROI_lvl_control = [f'{ROI}_BOLD' for ROI in ROIs_match]

def get_bounding_box(mask):
    x, y, z = np.where(mask)

    return min(x), max(x), min(y), max(y), min(z), max(z)

def handle_irregular_3d_fft(data, fill_value=0):
    """
    Perform 3D Fourier Transform on irregular data with NaN values

    Parameters:
    data (ndarray): 3D input array with NaN values representing void space
    fill_value (float): Value to replace NaNs with (default 0)

    Returns:
    ndarray: Fourier transformed data
    ndarray: Mask of valid data points
    ndarray: Frequency spectrum (magnitude)
    """
    # Create mask of valid (non-NaN) points
    mask = ~np.isnan(data)

    # Create a copy of the data and fill NaNs
    filled_data = np.copy(data)
    filled_data[~mask] = fill_value

    # Perform 3D FFT on filled data
    fft_data = np.fft.fftn(filled_data)
    fft_shifted = np.fft.fftshift(fft_data)
    magnitude_spectrum = np.abs(fft_shifted)

    return fft_shifted, mask, magnitude_spectrum

def get_img_box(region, fp, sn):
    # print(region)
    # quit()

    if '_L' in region:
        L_R = 'L'
        region = region.replace('_L', '')
    elif '_R' in region:
        L_R = 'R'
        region = region.replace('_R', '')
    else:
        L_R = None

    df_sn = get_trial_info(sn, easy_override=False, verbose=-1)
    sess = (fp.split('_')[0].replace('2', '').replace('3', '').replace('4', '').
            replace('7', '').replace('8', ''))
    df_sn.sort_values(by=f'{sess}_trial', inplace=True)

    img, good_idxs = utils.load_ni_w_nan_fps(df_sn[fp])

    ROIs = get_ROIs_from_region(region)
    if L_R:
        ROIs = [ROI for ROI in ROIs if f'_{L_R}_' in ROI]

    ROI_idxs = [int(roi.split(' ')[0]) for roi in ROIs]

    atlas = get_atlas(combine_regions=False, combine_bilateral=False,)
    atlas_data = atlas['maps'].get_fdata()
    mask = np.zeros(atlas_data.shape)
    for idx in ROI_idxs:
        mask[atlas_data == idx] = 1
    min_x, max_x, min_y, max_y, min_z, max_z = get_bounding_box(mask)
    img[mask < 0.5, :] = np.nan

    img_box = img[min_x:max_x, min_y:max_y, min_z:max_z, :]
    return img_box

def get_ffts(sn, region, fp, odd_only=True):
    img_box = get_img_box(region, fp, sn)
    if odd_only:
        if img_box.shape[0] % 2 == 0:
            new_mat = np.full((1, img_box.shape[1], img_box.shape[2],
                               img_box.shape[3]), np.nan)
            img_box = np.concatenate((img_box, new_mat), axis=0)
        if img_box.shape[1] % 2 == 0:
            new_mat = np.full((img_box.shape[0], 1, img_box.shape[2],
                               img_box.shape[3]), np.nan)
            img_box = np.concatenate((img_box, new_mat), axis=1)
        if img_box.shape[2] % 2 == 0:
            new_mat = np.full((img_box.shape[0], img_box.shape[1], 1,
                               img_box.shape[3]), np.nan)
            img_box = np.concatenate((img_box, new_mat), axis=2)

    ffts = np.empty_like(img_box, dtype=np.complex64)
    magnitudes = np.empty_like(img_box)
    for trial in range(114):
        fft_shifted, mask, magnitude_spectrum = (
            handle_irregular_3d_fft(img_box[..., trial]))
        ffts[..., trial] = fft_shifted

        magnitudes[..., trial] = magnitude_spectrum
    mask = ~np.isnan(img_box[..., 0])
    return ffts, mask


def extract_fz(ffts, fz, eight_corners=False):
    midpoint = np.array(ffts.shape[:-1]) // 2
    # midpoint += np.array([1, 1, 1])

    low_x0 = midpoint[0] - fz[1] + 1
    low_x1 = midpoint[0] - fz[0] + 1
    high_x0 = midpoint[0] + fz[0]
    high_x1 = midpoint[0] + fz[1]

    low_y0 = midpoint[1] - fz[1] + 1
    low_y1 = midpoint[1] - fz[0] + 1
    high_y0 = midpoint[1] + fz[0]
    high_y1 = midpoint[1] + fz[1]

    low_z0 = midpoint[2] - fz[1] + 1
    low_z1 = midpoint[2] - fz[0] + 1
    high_z0 = midpoint[2] + fz[0]
    high_z1 = midpoint[2] + fz[1]

    if eight_corners:
        oct0 = ffts[low_x0:low_x1, low_y0:low_y1, low_z0:low_z1, :]
        oct1 = ffts[low_x0:low_x1, low_y0:low_y1, high_z0:high_z1, :]
        oct2 = ffts[low_x0:low_x1, high_y0:high_y1, low_z0:low_z1, :]
        oct3 = ffts[low_x0:low_x1, high_y0:high_y1, high_z0:high_z1, :]
        oct4 = ffts[high_x0:high_x1, low_y0:low_y1, low_z0:low_z1, :]
        oct5 = ffts[high_x0:high_x1, low_y0:low_y1, high_z0:high_z1, :]
        oct6 = ffts[high_x0:high_x1, high_y0:high_y1, low_z0:low_z1, :]
        oct7 = ffts[high_x0:high_x1, high_y0:high_y1, high_z0:high_z1, :]
        octs = [oct0, oct1, oct2, oct3, oct4, oct5, oct6, oct7]
    else:
        # These don't overlap at all
        oct0 = ffts[low_x0:low_x1, low_y0:high_y1, low_z0:high_z1, :]
        oct1 = ffts[high_x0:high_x1, low_y0:high_y1, low_z0:high_z1, :]
        oct2 = ffts[low_x0:high_x1, low_y0:low_y1, low_z0:high_z1, :]
        oct3 = ffts[low_x0:high_x1, high_y0:high_y1, low_z0:high_z1, :]
        oct4 = ffts[low_x0:high_x1, low_y0:high_y1, low_z0:low_z1, :]
        oct5 = ffts[low_x0:high_x1, low_y0:high_y1, high_z0:high_z1, :]
        octs = [oct0, oct1, oct2, oct3, oct4, oct5]
        octs = [np.reshape(oct, (-1, oct0.shape[-1])) for oct in octs]
        octs = np.array(octs)
    vals = np.reshape(octs, (-1, oct0.shape[-1])).T

    return vals


# test = np.random.normal(size=(49, 49, 49, 114))
# for i in range(1, 24):
#     print(i)
#     extract_fz(test, (i, i+1))
# quit()



def ffts_reconstruct(ffts, mask, fz, resize=None, eight_corners=False):
    assert ffts.shape[0] % 2 == 1
    assert ffts.shape[1] % 2 == 1
    assert ffts.shape[2] % 2 == 1


    midpoint = np.array(ffts.shape[:-1]) // 2
    # midpoint += np.array([1, 1, 1])

    low_x0 = midpoint[0] - fz[1] + 1
    low_x1 = midpoint[0] - fz[0] + 1
    high_x0 = midpoint[0] + fz[0]
    high_x1 = midpoint[0] + fz[1]

    low_y0 = midpoint[1] - fz[1] + 1
    low_y1 = midpoint[1] - fz[0] + 1
    high_y0 = midpoint[1] + fz[0]
    high_y1 = midpoint[1] + fz[1]

    low_z0 = midpoint[2] - fz[1] + 1
    low_z1 = midpoint[2] - fz[0] + 1
    high_z0 = midpoint[2] + fz[0]
    high_z1 = midpoint[2] + fz[1]

    mask_fz = np.zeros_like(ffts)
    if eight_corners:
        mask_fz[low_x0:low_x1, low_y0:low_y1, low_z0:low_z1, :] = 1
        mask_fz[low_x0:low_x1, low_y0:low_y1, high_z0:high_z1, :] = 1
        mask_fz[low_x0:low_x1, high_y0:high_y1, low_z0:low_z1, :] = 1
        mask_fz[low_x0:low_x1, high_y0:high_y1, high_z0:high_z1, :] = 1
        mask_fz[high_x0:high_x1, low_y0:low_y1, low_z0:low_z1, :] = 1
        mask_fz[high_x0:high_x1, low_y0:low_y1, high_z0:high_z1, :] = 1
        mask_fz[high_x0:high_x1, high_y0:high_y1, low_z0:low_z1, :] = 1
        mask_fz[high_x0:high_x1, high_y0:high_y1, high_z0:high_z1, :] = 1
    else:
        mask_fz[low_x0:low_x1, low_y0:high_y1, low_z0:high_z1, :] = 1
        mask_fz[high_x0:high_x1, low_y0:high_y1, low_z0:high_z1, :] = 1
        mask_fz[low_x0:high_x1, low_y0:low_y1, low_z0:high_z1, :] = 1
        mask_fz[low_x0:high_x1, high_y0:high_y1, low_z0:high_z1, :] = 1
        mask_fz[low_x0:high_x1, low_y0:high_y1, low_z0:low_z1, :] = 1
        mask_fz[low_x0:high_x1, low_y0:high_y1, high_z0:high_z1, :] = 1

    ffts_fz = ffts * mask_fz
    if resize:
        ffts_fz_ = np.empty(resize)
        for i in range(ffts.shape[-1]):
            ffts_fz_[..., i] = undo_interpolation(ffts_fz[..., i], resize[:-1])
        ffts_fz = ffts_fz_

    # reconstructeds = np.empty(ffts.shape)
    reconstructeds = []
    for i in range(ffts.shape[-1]):
        fft_unshifted = np.fft.ifftshift(ffts_fz[..., i])
        reconstructed = np.fft.ifftn(fft_unshifted)
        reconstructed = np.real(reconstructed)  # Take real part. Imaginary is near zero anyway
        reconstructeds.append(reconstructed[mask])

        # test = reconstructed[mask]
        # test2 = np.fft.ifftn(fft_unshifted)[mask]
        # r, p = stats.spearmanr(test, test2)
        # print(f'{r=:.4f}')
    reconstructeds = np.array(reconstructeds)
    return reconstructeds

# ffts_reconstruct(np.random.normal(size=(10, 20, 15, 114)), None,(0, 1))

def get_uniform_size_ffts(sn, region, fp):
    ffts, mask = utils.pickle_wrap(get_ffts, None,
                                   kwargs={'sn': sn, 'region': region,
                                           'fp': fp, },
                                   verbose=-1, easy_override=False)
    orig_size = ffts.shape
    # t_st = time()
    ffts_ = np.empty((49, 49, 49, ffts.shape[-1]))
    for i in range(ffts.shape[-1]):
        ffts_[..., i] = interpolate_to_size(ffts[..., i], size=49)
    ffts = np.array(ffts_)
    return ffts, mask, orig_size
    # t_end = time()


def run_FFT_RSA(sn, region, fp, fz=(1, 5), semantic=True, dnn_layer=None,
                do_mag=False, do_reconstruct=False, uniform_size=True,
                eight_corners=False):
    assert not (do_mag and do_reconstruct)

    if uniform_size:
        ffts, mask, orig_size = utils.pickle_wrap(get_uniform_size_ffts, None,
                                       kwargs={'sn': sn, 'region': region,
                                               'fp': fp, },
                                       verbose=-1, easy_override=False)
    else:
        ffts, mask = utils.pickle_wrap(get_ffts, None,
                                 kwargs={'sn': sn, 'region': region,
                                         'fp': fp,},
                                 verbose=-1, easy_override=False)
        orig_size = ffts.shape

    # print(ffts.shape
    # if uniform_size:
    #     t_st = time()
    #     ffts_ = np.empty((49, 49, 49, ffts.shape[-1]))
    #     for i in range(ffts.shape[-1]):
    #         ffts_[..., i] = interpolate_to_size(ffts[..., i], size=49)
    #     ffts = np.array(ffts_)
    #     t_end = time()
    #     print(f'{t_end - t_st=:.2f}')
    #     quit()

    if do_mag:
        ffts = np.abs(ffts)
    if do_reconstruct:
        fp_reconstruct = fr'cache/fft_reconstruct/{sn}_{region}_{fp}_{fz}_{uniform_size}.pkl'
        # eight_corners = False
        kw = {'ffts': ffts, 'mask': mask, 'fz': fz,
              'eight_corners': eight_corners}
        if uniform_size:
            kw['resize'] = orig_size
        vals = utils.pickle_wrap(ffts_reconstruct, fp_reconstruct,
                                 kwargs=kw,
                                 verbose=-1, easy_override=False)
        neural_RSM = np.corrcoef(vals)
        # print(neural_RSM)
        # quit()
    else:
        # t_st = time()
        vals = extract_fz(ffts, fz, eight_corners=eight_corners)
        neural_RSM = np.corrcoef(vals)
        neural_RSM = np.real(neural_RSM)
        # t_end = time()
        # print(f'{t_end - t_st=:.2f}')

    neural_RSM[np.diag_indices_from(neural_RSM)] = np.nan
    trils = np.tril_indices_from(neural_RSM, k=-1)
    neural_RSM = neural_RSM[trils]

    stim_RSM = get_sn_fp_stim_RSM(sn, fp, semantic=semantic,
                                  dnn_layer=dnn_layer)
    stim_RSM = stim_RSM[trils]

    r, _ = stats.spearmanr(neural_RSM, stim_RSM, nan_policy='omit')
    return r

def run_all_sns(region='IT', fz=(1, 5), semantic=True, dnn_layer=None,
                do_reconstruct=False, eight_corners=False):
    sns = get_sns('all')['healthy']
    bad_sns = ['116', '125', '133', '213', '215', '231']
    sns = [sn for sn in sns if sn not in bad_sns]
    fps = ['bl7_fMRI', 'obj7_fMRI', 'con7_fMRI', 'vis7_fMRI']
    # fps = ['bl7_fMRI', 'obj7_fMRI', 'vis7_fMRI']
    # fps = ['bl7_fMRI']
    rs_all = np.full((len(sns), len(fps)), np.nan)
    # print(f'GO: {region}, {fz}')
    sns = sns[::-1]
    for i, sn in enumerate(sns):
        for j, fp in enumerate(fps):
            # print('test')
            # quit()
            r = utils.pickle_wrap(run_FFT_RSA, None,
                                  kwargs={'sn': sn, 'region': region,
                                          'fp': fp, 'fz': fz,
                                          'semantic': semantic,
                                          'dnn_layer': dnn_layer,
                                          'do_mag': False,
                                          'do_reconstruct': do_reconstruct,
                                          'uniform_size': True,
                                          'eight_corners': eight_corners},
                                  easy_override=True,
                                  verbose=-1,)
            rs_all[i, j] = r
    assert np.sum(np.isnan(rs_all)) == 0
    subj_rs = np.mean(rs_all, axis=1)
    t, p = stats.ttest_1samp(subj_rs, 0)
    M = np.mean(subj_rs)
    SE = stats.sem(subj_rs)
    print(f'{region} : {fz} | {M=:.2f}, {SE=:.2f}, {t=:.2f}, {p=:.3f}')
    return subj_rs

def analyze_multi_fz(region='IT', eight_corners=False,
                     do_reconstruct=False):
    print(f'Semantic')

    # fzs = [(i, i+3) for i in range(1, 24, 3)]
    fzs = [(i, i+1) for i in range(1, 24, 1)]

    for fz in fzs:
        run_all_sns(region, fz, True, None,
                    do_reconstruct=do_reconstruct,
                    eight_corners=eight_corners)
    #
    # print('Perceptual: DNN = 2')
    # # fzs = [(1, 4), (4, 8), (8, 12), ]
    # for fz in fzs:
    #     run_all_sns(region, fz, False, 2)

    # print('Perceptual: DNN = 6')
    # fzs = [(1, 4), (4, 8), (8, 12), ]
    # for fz in fzs:
    #     run_all_sns(region, fz, False, 6)

    for DNN_layer in range(2, 32, 2):
        print(f'Perceptual: DNN = {DNN_layer}')
        for fz in fzs:
            run_all_sns(region, fz, False, DNN_layer,
                        do_reconstruct=do_reconstruct,
                        eight_corners=eight_corners)

def plot_fz_interaction():
    pass

def interpolate_to_size(ffts, size=49):
    Nx, Ny, Nz = ffts.shape[:3]
    kx_orig = np.fft.fftfreq(Nx)
    kx_orig = np.fft.fftshift(kx_orig)
    ky_orig = np.fft.fftfreq(Ny)
    ky_orig = np.fft.fftshift(ky_orig)
    kz_orig = np.fft.fftfreq(Nz)
    kz_orig = np.fft.fftshift(kz_orig)

    if size:
        k_uniform = np.fft.fftfreq(size)
    else:
        k_uniform = np.fft.fftfreq(np.max([Nx, Ny, Nz]))

    k_uniform = np.fft.fftshift(k_uniform)

    interpolator_real = RegularGridInterpolator(
        (kx_orig, ky_orig, kz_orig),
        ffts.real,
        bounds_error=False,
        fill_value=0
    )

    interpolator_imag = RegularGridInterpolator(
        (kx_orig, ky_orig, kz_orig),
        ffts.imag,
        bounds_error=False,
        fill_value=0
    )

    KX, KY, KZ = np.meshgrid(k_uniform, k_uniform, k_uniform, indexing='ij')
    points = np.column_stack((KX.ravel(), KY.ravel(), KZ.ravel()))

    ffts_uniform_real = interpolator_real(points).reshape(KX.shape)
    ffts_uniform_imag = interpolator_imag(points).reshape(KX.shape)
    ffts_uniform = ffts_uniform_real + 1j * ffts_uniform_imag
    return ffts_uniform

def undo_interpolation(ffts, target_size):
    Nx, Ny, Nz = ffts.shape
    kx_orig = np.fft.fftfreq(Nx)
    kx_orig = np.fft.fftshift(kx_orig)
    ky_orig = np.fft.fftfreq(Ny)
    ky_orig = np.fft.fftshift(ky_orig)
    kz_orig = np.fft.fftfreq(Nz)
    kz_orig = np.fft.fftshift(kz_orig)

    kx_target = np.fft.fftfreq(target_size[0])
    kx_target = np.fft.fftshift(kx_target)
    ky_target = np.fft.fftfreq(target_size[1])
    ky_target = np.fft.fftshift(ky_target)
    kz_target = np.fft.fftfreq(target_size[2])
    kz_target = np.fft.fftshift(kz_target)

    # k_uniform = np.fft.fftfreq(target_size)
    # k_uniform = np.fft.fftshift(k_uniform)

    interpolator_real = RegularGridInterpolator(
        (kx_orig, ky_orig, kz_orig),
        ffts.real,
        bounds_error=False,
        fill_value=0
    )

    interpolator_imag = RegularGridInterpolator(
        (kx_orig, ky_orig, kz_orig),
        ffts.imag,
        bounds_error=False,
        fill_value=0
    )

    KX, KY, KZ = np.meshgrid(kx_target, ky_target, kz_target, indexing='ij')
    points = np.column_stack((KX.ravel(), KY.ravel(), KZ.ravel()))

    ffts_uniform_real = interpolator_real(points).reshape(KX.shape)
    ffts_uniform_imag = interpolator_imag(points).reshape(KX.shape)
    ffts_uniform = ffts_uniform_real + 1j * ffts_uniform_imag
    # print(ffts_uniform.shape)
    # quit()
    return ffts_uniform

# interpolate_to_largest(np.random.normal(size=(10, 20, 15)))

if __name__ == '__main__':
    EIGHT_CORNERS = False
    DO_RECONSTRUCT = False
    # analyze_multi_fz('IT_L', do_reconstruct=DO_RECONSTRUCT,
    #                  eight_corners=EIGHT_CORNERS)
    # analyze_multi_fz('IT_R', do_reconstruct=DO_RECONSTRUCT,
    #                  eight_corners=EIGHT_CORNERS)
    analyze_multi_fz('Occipital_R', do_reconstruct=DO_RECONSTRUCT,
                     eight_corners=EIGHT_CORNERS)
    analyze_multi_fz('Occipital_L', do_reconstruct=DO_RECONSTRUCT,
                     eight_corners=EIGHT_CORNERS)



    # analyze_multi_fz('Occipital_R')
    # analyze_multi_fz('ITL_L')
    # analyze_multi_fz('ITL_R')

    # analyze_multi_fz('ITL')
    # analyze_multi_fz('OC_T')
    # analyze_multi_fz('Temporal')
    # analyze_multi_fz('OC_IT')

    # IT_R: (29, 45, 27, 114)
    # Occipital_R: (29, 33, 37, 114)
