import os
import pathlib

path = pathlib.Path(__file__).parent.parent.resolve()
os.chdir(path)

import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt
from tqdm import tqdm
import mne

from Study3.analyze_plot_Fig7 import get_hrf


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


def conv_quick(ts, tr=2):
    import scipy.ndimage as ndimage
    HRF = get_hrf(tr)

    ts_ = ndimage.convolve1d(ts, HRF, mode='nearest',
                             origin=-HRF.shape[0] // 2, axis=0)
    return ts_


def setup_x(time):
    t_ticks = list(range(0, time + 1, 2))
    plt.xlim(0, time)
    plt.grid(axis='x', linestyle='--', alpha=0.9, which='minor')
    plt.xticks(t_ticks, minor=True, )
    plt.gca().tick_params(axis='x', which='minor', length=0)
    TR_ticks = list(range(1, time + 1, 2))
    TR_strs = ['TR$_{' f'{tick // 2 + 1:.0f}' '}$' for tick in TR_ticks]
    plt.xticks(TR_ticks, TR_strs, minor=False, fontsize=9)


def plot_EEG_x_fMRI():
    np.random.seed(0)
    true_resolution = 100
    time = 20
    fMRI_TR = 2
    amplitude_autocorr = 0.995
    true_hz = 3

    ts = np.linspace(0, time, time * true_resolution)

    fig, axs = plt.subplots(5, 2, figsize=(6.5, 8),
                            constrained_layout=True)

    osc = np.sin(np.pi * ts * true_hz)
    plt.sca(axs[0, 0])
    plt.title(f'{true_hz} Hz oscillation')
    plt.plot(ts, osc, linewidth=0.5)
    plt.xticks([0, 5, 10, 15, 20], ['0 s', '5 s', '10 s', '15 s', '20 s'])
    plt.xlim(0, 20)

    amplitude = sample_signal(len(ts), amplitude_autocorr)
    amplitude = np.abs(amplitude)
    plt.sca(axs[0, 1])
    plt.title('Amplitude')
    plt.plot(ts, amplitude, linewidth=0.5, color='green')
    plt.xticks([0, 5, 10, 15, 20], ['0 s', '5 s', '10 s', '15 s', '20 s'])
    plt.xlim(0, 20)
    plt.ylim(0, np.max(amplitude) * 1.1)

    gs = axs[0, 0].get_gridspec()
    for ax in axs[1, :]:
        ax.remove()
    axbig = fig.add_subplot(gs[1, :])
    osc_a = osc * amplitude

    plt.sca(axbig)
    plt.title('Oscillation x amplitude')
    plt.plot(ts, osc_a, linewidth=0.5, color='teal')
    plt.xticks([0, 5, 10, 15, 20], ['0 s', '5 s', '10 s', '15 s', '20 s'])
    plt.xlim(0, 20)

    osc_a *= -1

    VD = osc_a
    PA = -osc_a

    plt.sca(axs[2, 0])
    plt.title('True PA & VD signals')
    plt.plot(ts, PA, linewidth=0.5, color='blue', label='PA')
    plt.plot(ts, VD, linewidth=0.5, color='red', label='VD')
    l = plt.legend(frameon=False, loc=(0.505, 0.73), ncol=2,
                   columnspacing=0.6, handlelength=1, fontsize=12,
                   handletextpad=0.4)
    l.get_texts()[0].set_color('blue')
    l.legend_handles[0].set_linewidth(1.05)
    l.get_texts()[1].set_color('red')
    l.legend_handles[1].set_linewidth(1.05)
    setup_x(time)

    true_bins_TR = fMRI_TR * true_resolution
    true_bins_TR = int(true_bins_TR)

    ts_ = ts.reshape(-1, true_bins_TR).mean(axis=1)
    VD_ = VD.reshape(-1, true_bins_TR).mean(axis=1)
    PA_ = PA.reshape(-1, true_bins_TR).mean(axis=1)

    plt.sca(axs[3, 0])
    plt.title('Measured PA & VD signals')
    for i in range(len(ts_)):
        t = ts_[i]
        plt.plot([t, t], [VD_[i], PA_[i]], color='purple', linewidth=1,
                 zorder=-1, linestyle='--')
    plt.scatter(ts_, VD_, linewidth=0.5, color='red', label='VD')
    plt.scatter(ts_, PA_, linewidth=0.5, color='blue', label='PA')

    plt.text(16, 0.1, 'PA', color='blue', ha='center', fontsize=12, va='center')
    plt.text(16, -0.117, 'VD', color='red', ha='center', fontsize=12,
             va='center')

    plt.ylim(-.18, .18)
    setup_x(time)

    plt.sca(axs[4, 0])
    plt.title(f'|PA - VD| difference')
    ds = np.abs(VD_ - PA_)
    plt.scatter(ts_, ds, linewidth=0.5, color='purple')
    plt.ylim(0, 0.33)
    plt.yticks([0, 0.1, 0.2, 0.3])
    setup_x(time)

    tfr_in = osc_a[None, None, :]
    freqs = [0.5, 3, 10]
    tfr = mne.time_frequency.tfr_array_morlet(tfr_in,
                                              sfreq=true_resolution,
                                              freqs=freqs,
                                              n_cycles=3,
                                              zero_mean=True,
                                              output='power')[0, 0]
    tfr = np.sqrt(tfr)
    tfr1 = tfr[0]
    plt.sca(axs[2, 1])
    plt.title(f'{freqs[0]} Hz power')
    plt.plot(ts[true_resolution * 2:], tfr1[true_resolution * 2:],
             linewidth=0.5, color='limegreen')
    tfr1_ = tfr1.reshape(-1, true_bins_TR).mean(axis=1)
    plt.scatter(ts_[1:], tfr1_[1:], color='darkgreen', alpha=.9, zorder=2)
    plt.ylim(0, np.max(tfr1) * 1.05)
    setup_x(time)
    r, p = stats.pearsonr(tfr1_, ds)
    print(f'{r=:.4f}')

    tfr2 = tfr[1]
    plt.sca(axs[3, 1])
    plt.title(f'{freqs[1]} Hz power')
    plt.plot(ts, tfr2, linewidth=0.5, color='limegreen')
    tfr2_ = tfr2.reshape(-1, true_bins_TR).mean(axis=1)
    plt.scatter(ts_, tfr2_, color='darkgreen', alpha=.9, zorder=2)
    plt.ylim(0, np.max(tfr2) * 1.05)
    setup_x(time)
    r, p = stats.pearsonr(tfr2_, ds)
    print(f'{r=:.4f}')

    tfr3 = tfr[2]
    plt.sca(axs[4, 1])
    plt.title(f'{freqs[2]} Hz power')
    plt.plot(ts, tfr3, linewidth=0.5, color='limegreen')
    tfr3_ = tfr3.reshape(-1, true_bins_TR).mean(axis=1)
    plt.scatter(ts_, tfr3_, color='darkgreen', alpha=.9, zorder=2)
    plt.ylim(0, np.max(tfr3) * 1.05)
    setup_x(time)
    r, p = stats.pearsonr(tfr3_, ds)
    print(f'{r=:.4f}')

    fig.canvas.draw()

    axs = [axs[0, 0], axs[0, 1],
           axbig,
           axs[2, 0], axs[3, 0],
           axs[4, 0], axs[2, 1], axs[3, 1], axs[4, 1]]

    labels = ['(A)', '(B)', '(C)', '(D)', '(E)', '(F)', '(G)', '(H)', '(I)']
    from matplotlib.transforms import ScaledTranslation

    for ax, label in zip(axs, labels):
        try:
            bbox = ax.get_tightbbox(fig.canvas.get_renderer())
        except AttributeError:
            continue
        ax.text(
            0.0, 1.0, label, transform=(
                    ax.transAxes + ScaledTranslation(-20 / 72, +7 / 72,
                                                     fig.dpi_scale_trans)),
            va='bottom', fontweight="bold", fontsize=12)

    fp_out = fr'result_pics/SuppMat_EEG_fMRI_sim.png'
    plt.savefig(fp_out, dpi=600)
    plt.show()


def do_EEG_fMRI_sim(noise_mag=0.1, verbose=1, n_cycles=7, time=60_000):
    true_resolution = 100

    fMRI_TR = 2
    amplitude_autocorr = 0.95
    true_hz = 3

    amplitude = sample_signal(time * true_resolution, amplitude_autocorr,
                              0, 1)
    amplitude = np.abs(amplitude)

    noise_mag = noise_mag
    noise = sample_signal(time * true_resolution, amplitude_autocorr,
                          0, noise_mag)
    noise2 = sample_signal(time * true_resolution, amplitude_autocorr,
                           0, noise_mag)
    noise_eeg = sample_signal(time * true_resolution, amplitude_autocorr,
                              0, noise_mag)

    osc = np.cos(np.pi * np.linspace(0, time, time * true_resolution) *
                 true_hz)
    osc_a = osc * amplitude
    inv_osc = -osc_a

    noisy_osc = osc_a + noise
    noisy_osc_inv = inv_osc + noise2

    true_bins_TR = fMRI_TR * true_resolution
    true_bins_TR = int(true_bins_TR)
    noisy_oscillator_TR = noisy_osc.reshape(-1, true_bins_TR).mean(
        axis=1)

    if verbose: print('Onto convolving')
    noisy_oscillator_fMRI = conv_quick(noisy_oscillator_TR)
    noisy_osc_inv_TR = noisy_osc_inv.reshape(-1, true_bins_TR).mean(axis=1)
    noisy_osc_inv_fMRI = conv_quick(noisy_osc_inv_TR)

    amplitude_TR = amplitude.reshape(-1, true_bins_TR).mean(axis=1)
    amplitude_fMRI = conv_quick(amplitude_TR)

    d = np.abs(noisy_oscillator_fMRI - noisy_osc_inv_fMRI)

    if verbose: print('Onto TFR')
    noisy_oscillator_e = osc_a + noise_eeg
    tfr_in = noisy_oscillator_e[None, None, :]
    tfr = mne.time_frequency.tfr_array_morlet(tfr_in,
                                              sfreq=true_resolution,
                                              freqs=np.arange(1, 31),
                                              n_cycles=n_cycles,
                                              zero_mean=True,
                                              output='power')[0, 0]

    r_amp_fMRI = stats.spearmanr(amplitude_fMRI, d)[0]
    print(F'{r_amp_fMRI=:.4f}')
    print(f'{tfr.shape=}')
    print(f'{n_cycles=} | {noise_mag=:.2f} | {amplitude_autocorr=}')

    for hz in np.arange(1, tfr.shape[0] + 1):
        r, p = stats.spearmanr(tfr[hz - 1], amplitude)
        amplitude_TR = amplitude.reshape(-1, true_bins_TR).mean(axis=1)
        amplitude_fMRI = conv_quick(amplitude_TR)
        tfr_TR = tfr[hz - 1].reshape(-1, true_bins_TR).mean(axis=1)

        tfr_fMRI = conv_quick(tfr_TR)

        r_af, p = stats.spearmanr(tfr_fMRI, amplitude_fMRI)

        r_f, p = stats.spearmanr(tfr_fMRI, d)

        if hz == true_hz:
            extra = ' ***'
        else:
            extra = ''

        print(f'{hz=}, {r=:.4f}, {r_af=:.4}, {r_f=:.4f}{extra}')


def do_multiple_EEG_fMRI_sim(noise_mag, nsims=100, time=2400):
    rs = []
    for _ in tqdm(range(nsims)):
        noise_mag = 0.5
        r = do_EEG_fMRI_sim(noise_mag=noise_mag, time=time)
        rs.append(r)
    rs = np.array(rs)
    M = np.mean(rs)
    SD = np.std(rs)
    SE = SD / np.sqrt(len(rs))
    print(f'{noise_mag:.2f} | {M=:.4f}, {SE=:.4f}')


if __name__ == '__main__':
    plot_EEG_x_fMRI()
    do_EEG_fMRI_sim()
