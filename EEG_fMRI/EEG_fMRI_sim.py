import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt
from tqdm import tqdm
import mne

from EEG_fMRI.EEG_fMRI_test import get_hrf

def sample_signal(n_samples, corr, mu=0, sigma=1):
    assert 0 < corr < 1, "Auto-correlation must be between 0 and 1"

    # Find out the offset `c` and the std of the white noise `sigma_e`
    # that produce a signal with the desired mean and variance.
    # See https://en.wikipedia.org/wiki/Autoregressive_model
    # under section "Example: An AR(1) process".
    c = mu * (1 - corr)
    sigma_e = np.sqrt((sigma ** 2) * (1 - corr ** 2))

    # Sample the auto-regressive process.
    signal = [c + np.random.normal(0, sigma_e)]
    for _ in range(1, n_samples):
        signal.append(c + corr * signal[-1] + np.random.normal(0, sigma_e))

    return np.array(signal)

def conv_quick(ts):
    import scipy.ndimage as ndimage
    HRF = get_hrf()

    ts_ = ndimage.convolve1d(ts, HRF, mode='nearest',
                                  origin=-HRF.shape[0] // 2, axis=0)
    return ts_

def do_EEG_fMRI_sim(noise_mag=0.1):
    true_resolution = 250
    time = 6000
    fMRI_TR = 2
    amplitude_autocorr = 0.99
    true_hz = 10

    amplitude = sample_signal(time * true_resolution, amplitude_autocorr,
                              0, 1)
    amplitude = np.abs(amplitude)

    noise_mag = noise_mag
    noise = sample_signal(time * true_resolution, amplitude_autocorr,
                          0, noise_mag)
    noise2 = sample_signal(time * true_resolution, amplitude_autocorr,
                           0, noise_mag)

    oscillator = np.cos(2 * np.pi *
                        np.linspace(0, time, time * true_resolution) *
                        true_hz)
    oscillator *= amplitude
    inv_osc = -oscillator

    # TODO FFT

    noisy_oscillator = oscillator + noise
    noisy_osc_inv = inv_osc + noise2

    true_bins_TR = fMRI_TR * true_resolution
    noisy_oscillator_TR = noisy_oscillator.reshape(-1, true_bins_TR).mean(
        axis=1)
    noisy_oscillator_fMRI = conv_quick(noisy_oscillator_TR)
    noisy_osc_inv_TR = noisy_osc_inv.reshape(-1, true_bins_TR).mean(axis=1)
    noisy_osc_inv_fMRI = conv_quick(noisy_osc_inv_TR)

    amplitude_TR = amplitude.reshape(-1, true_bins_TR).mean(axis=1)
    amplitude_fMRI = conv_quick(amplitude_TR)

    # d = np.abs(np.fft.fft(noisy_oscillator_fMRI))
    d = np.abs(noisy_oscillator_fMRI - noisy_osc_inv_fMRI)
    r, p = stats.spearmanr(d, amplitude_fMRI)

    tfr_in = noisy_oscillator[None, None, :]
    tfr = mne.time_frequency.tfr_array_morlet(tfr_in,
                                              sfreq=true_resolution,
                                              freqs = np.arange(1, 51),
                                              n_cycles=1,
                                              output='power')[0, 0]
    for hz in np.arange(1, 51):
        r, p = stats.pearsonr(tfr[hz - 1], amplitude)
        print(f'{hz=}, {r=:.4f}')
    quit()


    # plt.plot(tfr[30 - 1], linewidth=0.5)
    # # plt.plot(oscillator)
    # plt.plot(amplitude)
    # r_test, _ = stats.pearsonr(tfr[true_hz - 1], amplitude)
    # plt.title(f'{r_test=:.3f}')
    # plt.show()
    # # print(tfr)
    # print(tfr.shape)
    # quit()

    # plt.plot(oscillator, linewidth=0.5)
    # plt.show()
    # print(f'{r=:.4f}')
    return r


def do_multiple_EEG_fMRI_sim(noise_mag, nsims=100):
    rs = []
    for nsim in tqdm(range(nsims)):
        noise_mag = 1
        r = do_EEG_fMRI_sim(noise_mag=noise_mag)
        rs.append(r)
    rs = np.array(rs)
    M = np.mean(rs)
    SD = np.std(rs)
    SE = SD / np.sqrt(len(rs))
    print(f'{noise_mag:.2f} | {M=:.4f}, {SE=:.4f}')

if __name__ == '__main__':
    for nm in np.logspace(-2, 2, 10, base=10):
        do_multiple_EEG_fMRI_sim(nm)
