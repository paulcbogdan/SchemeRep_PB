import numpy as np
from scipy.interpolate import RegularGridInterpolator

import utils
from Utils.atlas_funcs import get_BNA_ROIs, get_atlas
from connRSA.DistRep_ROI_RSA import ROI2NETWORK
from networks.old.networks import prep_networks
from organize_bhv import get_trial_info


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
    if len(fz) == 3 and fz[2] == 'all':
        mask_fz[low_x0:low_x1, :, :, :] = 1
        mask_fz[high_x0:high_x1, :, :, :] = 1
        mask_fz[:, low_y0:low_y1, :, :] = 1
        mask_fz[:, high_y0:high_y1, :, :] = 1
        mask_fz[:, :, low_z0:low_z1, :] = 1
        mask_fz[:, :, high_z0:high_z1, :] = 1
    elif eight_corners:
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


def get_uniform_size_ffts(sn, region, fp):
    ffts, mask = utils.pickle_wrap(get_ffts, None,
                                   kwargs={'sn': sn, 'region': region,
                                           'fp': fp, },
                                   verbose=-1, easy_override=False)
    orig_size = ffts.shape

    if region[:5] == 'OC_IT':
        big_size = 67
    else:
        big_size = 49

    ffts_ = np.empty((big_size, big_size, big_size, ffts.shape[-1]))
    for i in range(ffts.shape[-1]):
        ffts_[..., i] = interpolate_to_size(ffts[..., i], size=big_size)
    ffts = np.array(ffts_)
    return ffts, mask, orig_size
    # t_end = time()


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
