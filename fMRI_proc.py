import numpy as np

from utils import stdize
import scipy.stats as stats

from warnings import filterwarnings

filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
filterwarnings('ignore', category=RuntimeWarning,
               message='Degrees of freedom <= 0')
filterwarnings('ignore', category=RuntimeWarning,
               message='NaNs of infinite values are')

def within_run_to_nan(RDM):
    RDM_ = RDM.copy()
    trial_per_run = RDM.shape[0] // 3
    for run in range(3):
        low = run * trial_per_run
        high = (run + 1) * trial_per_run
        RDM_[low:high, low:high] = np.nan
    # TODO: Fix, this won't work properly except for on encoding!!
    return RDM_

def RDM_x_RDM_by_run(fMRI_RDM, RSM_stim, corr='spear'):
    RSM_stim[np.diag_indices_from(RSM_stim)] = np.nan
    fMRI_RDM[np.diag_indices_from(fMRI_RDM)] = np.nan
    zs = []
    trial_per_run = RSM_stim.shape[0] // 3
    for run0 in range(3):
        for run1 in range(3):
            if run1 < run0:
                continue
            low0 = run0 * trial_per_run
            high0 = (run0 + 1) * trial_per_run
            low1 = run1 * trial_per_run
            high1 = (run1 + 1) * trial_per_run
            stim_flat = RSM_stim[low0:high0, low1:high1].flatten()
            fMRI_flat = fMRI_RDM[low0:high0, low1:high1].flatten()
            nans = np.isnan(stim_flat) | np.isnan(fMRI_flat)
            n_nans = np.sum(nans)
            assert n_nans == 0 or n_nans == 38, f'RDM x RDM bad nans: {n_nans=}'
            stim_flat = stim_flat[~nans]
            fMRI_flat = fMRI_flat[~nans]
            if corr == 'spear':
                # print(f'{fMRI_flat}')
                # print(f'{stim_flat}')
                r, _ = stats.spearmanr(fMRI_flat, stim_flat)
                z = np.arctanh(r)
            elif corr == 'corr':
                r, _ = stats.pearsonr(fMRI_flat, stim_flat)
                z = np.arctanh(r)
            elif corr == 'euc':
                z = -np.mean(fMRI_flat - stim_flat)
            else:
                raise KeyError(f'conn must be \"spear\", \"corr\", or \"euc\", not {corr}')
            zs.append(z)
            # print(f'{z=}')
    return np.nanmean(zs)

def RDM_x_RDM(fMRI_RDM, stim_RDM, corr='spear', within_to_nan=True):
    assert fMRI_RDM.shape == stim_RDM.shape, 'RDMs must be the same shape: ' \
       f'fMRI_RDM.shape = {fMRI_RDM.shape}, stim_RDM.shape = {stim_RDM.shape}'
    tril_idx = np.tril_indices_from(fMRI_RDM, k=-1)
    if within_to_nan:
        fMRI_RDM_ = within_run_to_nan(fMRI_RDM)
    else:
        fMRI_RDM_ = fMRI_RDM
    # plt.imshow(fMRI_RDM_)
    # plt.show()
    # quit()
    fMRI_flat = fMRI_RDM_[tril_idx]
    stim_flat = stim_RDM[tril_idx]
    nans = np.isnan(fMRI_flat) | np.isnan(stim_flat)
    fMRI_flat = fMRI_flat[~nans]
    stim_flat = stim_flat[~nans]
    if corr == 'spear':
        r, _ = stats.spearmanr(fMRI_flat, stim_flat)
        z = np.arctanh(r)
    elif corr == 'corr':
        r, _ = stats.pearsonr(fMRI_flat, stim_flat)
        z = np.arctanh(r)
    elif corr == 'euc':
        z = -np.mean(fMRI_flat - stim_flat)
    else:
        raise KeyError(f'conn must be \"spear\", \"corr\", or \"euc\", not {corr}')

    return z

def get_IRAFs(fMRI_RDM, stim_RDM, df_sn, within_to_nan=True,
              by_run=False, second_order='corr'):
    '''
    matmul all took 0.001 seconds
    matmul semi (one loop, inner matmul) took 0.008 seconds
    scipy pearsonr took 0.013 seconds
    scipy spearmanr took 0.055 seconds
    '''

    if within_to_nan:
        fMRI_RDM_ = within_run_to_nan(fMRI_RDM)
    elif by_run:
        fMRI_RDM_within_nan = within_run_to_nan(fMRI_RDM)
        fMRI_RDM_ = fMRI_RDM.copy()
        fMRI_RDM_[~np.isnan(fMRI_RDM_within_nan)] = np.nan
        # raise NotImplementedError
    else:
        fMRI_RDM_ = fMRI_RDM
    fMRI_RDM_[np.diag_indices_from(fMRI_RDM)] = np.nan
    stim_RDM[np.diag_indices_from(stim_RDM)] = np.nan
    stim_RDM_ = stim_RDM.copy()
    if second_order == 'corr':
        fMRI_RDM_std = stdize(fMRI_RDM_, axis=0, nans=True)
        stim_RDM_std = stdize(stim_RDM, axis=0, nans=True)
        IRAFs = np.nanmean(fMRI_RDM_std * stim_RDM_std, axis=0)
        IRAFs = np.arctanh(IRAFs)
    elif second_order == 'spear':
        fMRI_RDM_[np.isnan(stim_RDM_)] = np.nan
        stim_RDM_[np.isnan(fMRI_RDM_)] = np.nan
        fMRI_RDM_r = stats.rankdata(fMRI_RDM_, axis=0, nan_policy='omit')
        stim_RDM_r = stats.rankdata(stim_RDM_, axis=0, nan_policy='omit')
        fMRI_RDM_r = stdize(fMRI_RDM_r, axis=0, nans=True)
        stim_RDM_r = stdize(stim_RDM_r, axis=0, nans=True)
        IRAFs = np.nanmean(fMRI_RDM_r * stim_RDM_r, axis=0)
        IRAFs = np.arctanh(IRAFs)
    else:
        raise NotImplementedError(f'get_IRAFs {second_order=}')
    if df_sn is not None: IRAFs = IRAFs[df_sn['obj'].argsort()]
    return IRAFs




