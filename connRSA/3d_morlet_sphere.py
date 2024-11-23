import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def spherical_wavelet_3d(size=32, wavelength=8, sigma=1.0):
    """
    Create a spherically symmetric 3D wavelet.

    Parameters:
    -----------
    size : int
        Size of the wavelet grid
    wavelength : float
        Wavelength of radial oscillations
    sigma : float
        Width of Gaussian envelope
    """
    # Create coordinate grids
    x = np.linspace(-2, 2, size)
    y = np.linspace(-2, 2, size)
    z = np.linspace(-2, 2, size)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Calculate radial distance from center
    R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)

    # Create wavelet
    k = 2 * np.pi / wavelength
    gaussian = np.exp(-R ** 2 / (2 * sigma ** 2))
    oscillation = np.cos(k * R)  # Using real-valued oscillation for simplicity
    wavelet = gaussian * oscillation

    return wavelet, X, Y, Z, R


def visualize_spherical_wavelet(size=32, wavelength=8, sigma=1.0):
    """Visualize the spherical wavelet"""
    wavelet, X, Y, Z, R = spherical_wavelet_3d(size, wavelength, sigma)

    fig = plt.figure(figsize=(15, 5))

    # Plot 1: XY slice at middle Z
    ax1 = fig.add_subplot(131)
    mid_z = size // 2
    im1 = ax1.imshow(wavelet[:, :, mid_z].T,
                     extent=[-2, 2, -2, 2],
                     cmap='RdBu')
    ax1.set_title('XY Slice')
    plt.colorbar(im1, ax=ax1)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')

    # Plot 2: Radial profile
    ax2 = fig.add_subplot(132)
    # Get unique radial values and corresponding wavelet values
    unique_r = np.sort(np.unique(R))
    radial_profile = [np.mean(wavelet[R == r]) for r in unique_r]
    ax2.plot(unique_r, radial_profile)
    ax2.set_title('Radial Profile')
    ax2.set_xlabel('Radius')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True)

    # Plot 3: 3D visualization of selected isosurfaces
    ax3 = fig.add_subplot(133, projection='3d')

    # Create isosurfaces
    max_val = np.max(np.abs(wavelet))
    threshold = max_val * 0.3

    # Plot positive isosurface
    ax3.contour(X[:, :, size // 2],
                Y[:, :, size // 2],
                wavelet[:, :, size // 2],
                levels=[threshold],
                colors=['red'])

    # Plot negative isosurface
    ax3.contour(X[:, :, size // 2],
                Y[:, :, size // 2],
                wavelet[:, :, size // 2],
                levels=[-threshold],
                colors=['blue'])

    ax3.set_title('3D View')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')

    plt.tight_layout()
    return fig


def show_wavelet_examples():
    # Example 1: Basic spherical wavelet
    fig1 = visualize_spherical_wavelet(size=32, wavelength=8, sigma=2)
    fig1.suptitle('Spherical Wavelet (Standard)')

    # Example 2: Higher frequency wavelet
    fig2 = visualize_spherical_wavelet(size=32, wavelength=4, sigma=2)
    fig2.suptitle('Spherical Wavelet (Higher Frequency)')

    # Example 3: More spread out wavelet
    fig3 = visualize_spherical_wavelet(size=32, wavelength=8, sigma=1)
    fig3.suptitle('Spherical Wavelet (Larger Spread)')

    plt.show()


# Run the visualization
show_wavelet_examples()