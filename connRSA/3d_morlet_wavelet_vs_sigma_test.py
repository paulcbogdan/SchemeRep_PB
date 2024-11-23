import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def create_spherical_morlet_1d(x, wavelength, sigma):
    """Create 1D radial profile of spherical Morlet wavelet"""
    k = 2 * np.pi / wavelength
    gaussian = np.exp(-x ** 2 / (2 * sigma ** 2))
    oscillation = np.cos(k * x)
    return gaussian * oscillation


def visualize_wavelength_vs_sigma():
    """Visualize how wavelength and sigma affect the wavelet"""
    x = np.linspace(-5, 5, 1000)

    # Create figure with multiple comparisons
    fig = plt.figure(figsize=(15, 10))

    # Plot 1: Varying wavelength, fixed sigma
    ax1 = fig.add_subplot(221)
    wavelengths = [0.5, 1.0, 1000.0]
    sigma = 1.0
    for wavelength in wavelengths:
        wavelet = create_spherical_morlet_1d(x, wavelength, sigma)
        ax1.plot(x, wavelet, label=f'λ={wavelength}')

    ax1.set_title('Varying Wavelength (fixed σ=1.0)')
    ax1.legend()
    ax1.grid(True)
    ax1.set_xlabel('Radius')
    ax1.set_ylabel('Amplitude')

    # Plot 2: Varying sigma, fixed wavelength
    ax2 = fig.add_subplot(222)
    sigmas = [0.5, 1.0, 2.0]
    wavelength = 1.0
    for sigma in sigmas:
        wavelet = create_spherical_morlet_1d(x, wavelength, sigma)
        ax2.plot(x, wavelet, label=f'σ={sigma}')

    ax2.set_title('Varying Sigma (fixed λ=1.0)')
    ax2.legend()
    ax2.grid(True)
    ax2.set_xlabel('Radius')
    ax2.set_ylabel('Amplitude')

    # Plot 3: Gaussian envelope visualization
    ax3 = fig.add_subplot(223)
    sigma = 1.0
    wavelength = 1.0
    wavelet = create_spherical_morlet_1d(x, wavelength, sigma)
    envelope = np.exp(-x ** 2 / (2 * sigma ** 2))

    ax3.plot(x, wavelet, label='Wavelet')
    ax3.plot(x, envelope, 'r--', label='Gaussian envelope')
    ax3.plot(x, -envelope, 'r--')
    ax3.set_title('Wavelet with Gaussian Envelope')
    ax3.legend()
    ax3.grid(True)
    ax3.set_xlabel('Radius')
    ax3.set_ylabel('Amplitude')

    # Plot 4: Number of oscillations within envelope
    ax4 = fig.add_subplot(224)
    wavelengths = [0.5, 1.0]
    sigma = 1.0
    for wavelength in wavelengths:
        wavelet = create_spherical_morlet_1d(x, wavelength, sigma)
        envelope = np.exp(-x ** 2 / (2 * sigma ** 2))
        ax4.plot(x, wavelet, label=f'λ={wavelength}')

    ax4.plot(x, envelope, 'r--', label='Envelope (σ=1.0)')
    ax4.plot(x, -envelope, 'r--')
    ax4.set_title('Oscillations within Envelope')
    ax4.legend()
    ax4.grid(True)
    ax4.set_xlabel('Radius')
    ax4.set_ylabel('Amplitude')

    plt.tight_layout()
    return fig


# Create visualization
visualize_wavelength_vs_sigma()
plt.show()