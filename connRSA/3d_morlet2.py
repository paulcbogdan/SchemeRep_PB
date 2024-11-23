import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import convolve
import warnings


def create_spherical_morlet(wavelength, sigma, kernel_size=None):
    """
    Create a spherical Morlet wavelet kernel.

    Parameters:
    -----------
    wavelength : float
        Wavelength of the wavelet oscillations
    sigma : float
        Standard deviation of the Gaussian envelope
    kernel_size : int or None
        Size of the kernel. If None, calculated based on sigma

    Returns:
    --------
    kernel : ndarray
        3D array containing the Morlet wavelet
    """
    if kernel_size is None:
        # Make kernel large enough to capture 3 sigma
        kernel_size = int(2 * np.ceil(3 * sigma))
        # Make sure it's odd
        if kernel_size % 2 == 0:
            kernel_size += 1

    # Create coordinate grids
    x = np.linspace(-kernel_size // 2, kernel_size // 2, kernel_size)
    y = np.linspace(-kernel_size // 2, kernel_size // 2, kernel_size)
    z = np.linspace(-kernel_size // 2, kernel_size // 2, kernel_size)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Calculate radial distance
    R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)

    # Create wavelet
    k = 2 * np.pi / wavelength
    gaussian = np.exp(-R ** 2 / (2 * sigma ** 2))
    oscillation = np.exp(1j * k * R)
    kernel = gaussian * oscillation

    # Normalize
    kernel = kernel / np.sqrt(np.sum(np.abs(kernel) ** 2))

    return kernel


def analyze_with_morlet(data, wavelength, sigma, kernel_size=None):
    """
    Analyze 3D data using spherical Morlet wavelet.

    Parameters:
    -----------
    data : ndarray
        3D input data array (can contain NaN)
    wavelength : float
        Wavelength of the wavelet
    sigma : float
        Standard deviation of the Gaussian envelope
    kernel_size : int or None
        Size of the kernel. If None, calculated based on sigma

    Returns:
    --------
    magnitude : ndarray
        Magnitude of wavelet coefficients
    phase : ndarray
        Phase of wavelet coefficients
    """
    # Create mask for NaN values
    mask = np.isnan(data)

    # Fill NaN values with 0 for convolution
    data_filled = np.nan_to_num(data, nan=0.0)

    # Create wavelet kernel
    kernel = create_spherical_morlet(wavelength, sigma, kernel_size)

    # Perform convolution for real and imaginary parts
    coefficients = convolve(data_filled, np.real(kernel)) + \
                   1j * convolve(data_filled, np.imag(kernel))

    # Calculate magnitude and phase
    magnitude = np.abs(coefficients)
    phase = np.angle(coefficients)

    # Mask out NaN regions in result
    magnitude[mask] = np.nan
    phase[mask] = np.nan

    return magnitude, phase


def visualize_analysis(data, magnitude, phase, slice_idx=None):
    """
    Visualize original data and wavelet analysis results.

    Parameters:
    -----------
    data : ndarray
        Original 3D data
    magnitude : ndarray
        Magnitude of wavelet coefficients
    phase : ndarray
        Phase of wavelet coefficients
    slice_idx : tuple or None
        (x,y,z) indices for slices to show. If None, uses middle slices.
    """
    if slice_idx is None:
        slice_idx = (data.shape[0] // 2, data.shape[1] // 2, data.shape[2] // 2)

    fig = plt.figure(figsize=(15, 10))

    # magnitude = phase

    # Original data slices
    ax1 = fig.add_subplot(231)
    im1 = ax1.imshow(data[slice_idx[0], :, :], cmap='viridis')
    ax1.set_title('Original Data (YZ slice)')
    plt.colorbar(im1, ax=ax1)

    ax2 = fig.add_subplot(232)
    im2 = ax2.imshow(data[:, slice_idx[1], :], cmap='viridis')
    ax2.set_title('Original Data (XZ slice)')
    plt.colorbar(im2, ax=ax2)

    ax3 = fig.add_subplot(233)
    im3 = ax3.imshow(data[:, :, slice_idx[2]], cmap='viridis')
    ax3.set_title('Original Data (XY slice)')
    plt.colorbar(im3, ax=ax3)

    # Magnitude slices
    ax4 = fig.add_subplot(234)
    im4 = ax4.imshow(magnitude[slice_idx[0], :, :], cmap='magma')
    ax4.set_title('Wavelet Magnitude (YZ slice)')
    plt.colorbar(im4, ax=ax4)

    ax5 = fig.add_subplot(235)
    im5 = ax5.imshow(magnitude[:, slice_idx[1], :], cmap='magma')
    ax5.set_title('Wavelet Magnitude (XZ slice)')
    plt.colorbar(im5, ax=ax5)

    ax6 = fig.add_subplot(236)
    im6 = ax6.imshow(magnitude[:, :, slice_idx[2]], cmap='magma')
    ax6.set_title('Wavelet Magnitude (XY slice)')
    plt.colorbar(im6, ax=ax6)

    plt.tight_layout()
    return fig


# Example usage
def example_analysis():
    # Create sample data (a sphere with some noise)
    size = 50
    x = np.linspace(-2, 2, size)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)

    # Create a sphere with radius 1
    data = np.zeros_like(R)
    data[R <= 1] = 1

    # Add some noise
    data += 0.1 * np.random.randn(*data.shape)

    # Add some NaN values
    mask = np.random.rand(*data.shape) < 0.1
    data[mask] = np.nan

    # Analyze with Morlet wavelet
    wavelength = 50  # Should be appropriate for detecting the sphere
    sigma = 2
    magnitude, phase = analyze_with_morlet(data, wavelength, sigma)

    # Visualize results
    fig = visualize_analysis(data, magnitude, phase)
    plt.show()

    return data, magnitude, phase


if __name__ == "__main__":
    # Run example analysis
    data, magnitude, phase = example_analysis()