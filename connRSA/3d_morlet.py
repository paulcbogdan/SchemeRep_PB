import numpy as np
from scipy.fft import fftn, ifftn
import matplotlib.pyplot as plt

def morlet_wavelet_3d(size, wavelength, direction=(1, 0, 0), sigma=1.0):
    """
    Generate a 3D Morlet wavelet.

    Parameters:
    -----------
    size : tuple of int
        Size of the wavelet in voxels (nx, ny, nz)
    wavelength : float
        Wavelength of the wavelet oscillations
    direction : tuple
        Direction of wavelet oscillation (dx, dy, dz)
    sigma : float
        Standard deviation of the Gaussian envelope

    Returns:
    --------
    wavelet : ndarray
        Complex 3D array containing the Morlet wavelet
    """
    # Create coordinate grids
    x = np.linspace(-size[0] // 2, size[0] // 2, size[0])
    y = np.linspace(-size[1] // 2, size[1] // 2, size[1])
    z = np.linspace(-size[2] // 2, size[2] // 2, size[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Normalize direction vector
    direction = np.array(direction)
    direction = direction / np.linalg.norm(direction)

    # Calculate the projection of each point onto the direction vector
    projection = (X * direction[0] + Y * direction[1] + Z * direction[2])

    # Calculate the perpendicular distance from each point to the direction vector
    perpendicular_dist = np.sqrt(
        X ** 2 + Y ** 2 + Z ** 2 - projection ** 2
    )

    # Create the wavelet
    k = 2 * np.pi / wavelength
    gaussian = np.exp(-(perpendicular_dist ** 2 + projection ** 2) / (2 * sigma ** 2))
    oscillation = np.exp(1j * k * projection)

    wavelet = gaussian * oscillation

    # Normalize
    wavelet = wavelet / np.sqrt(np.sum(np.abs(wavelet) ** 2))

    return wavelet


def wavelet_transform_3d(data, wavelengths, directions, sigma=4.0):
    """
    Perform 3D continuous wavelet transform using Morlet wavelets.

    Parameters:
    -----------
    data : ndarray
        3D input data to be transformed
    wavelengths : list or array
        List of wavelengths to analyze
    directions : list of tuples
        List of directions to analyze
    sigma : float
        Standard deviation of the Gaussian envelope

    Returns:
    # --------
    coefficients : ndarray
        5D array of wavelet coefficients (wavelength, direction, x, y, z)
    """
    # Initialize output array
    coefficients = np.zeros((len(wavelengths), len(directions), *data.shape), dtype=complex)

    # Compute FFT of the data
    fft_data = fftn(data)

    # Loop over wavelengths and directions
    for i, wavelength in enumerate(wavelengths):
        for j, direction in enumerate(directions):
            # Create wavelet
            wavelet = morlet_wavelet_3d(data.shape, wavelength, direction, sigma)

            # plt.imshow(np.abs(wavelet[:, :, 16]))
            # plt.show()
            #
            # print(wavelet.shape)
            # quit()

            # Convolve in Fourier space
            coefficients[i, j] = ifftn(fft_data * fftn(wavelet))

    return coefficients


# Example usage
def example_usage():
    # Create sample data (e.g., a 3D Gaussian)
    size = (32, 32, 32)
    x = np.linspace(-4, 4, size[0])
    y = np.linspace(-4, 4, size[1])
    z = np.linspace(-4, 4, size[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    data = np.exp(-(X ** 2 + Y ** 2 + Z ** 2))

    plt.imshow(np.abs(data[:, 16, :]))
    plt.show()

    # Define analysis parameters
    wavelengths = [2, 4, 8]
    # wavelengths = [10]
    directions = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

    # Perform wavelet transform
    coefficients = wavelet_transform_3d(data, wavelengths, directions)
    # print(coefficients[:, :, 0, 2, 4])
    # print(coefficients.shape)
    # quit()

    plt.imshow(np.abs(coefficients[0, 0, :, 16, :]))
    plt.show()

    plt.imshow(np.abs(coefficients[1, 0, :, 16, :]))
    plt.show()

    plt.imshow(np.abs(coefficients[2, 0, :, 16, :]))
    plt.show()

    # plt.imshow(np.real(wavelet[0, 16, :]))
    # plt.colorbar()
    # plt.show()

    return coefficients


if __name__ == '__main__':
    example_usage()
    # wavelet = morlet_wavelet_3d((32, 32, 32), 8, (1, 1, 1), sigma=4.0)
    #
    # # print(wavelet)
    # plt.imshow(np.real(wavelet[:, 16, :]))
    # plt.colorbar()
    # plt.show()

    # example_usage()
    # test = morlet_wavelet_3d((50, 50, 50), 3, direction=(1, 0, 0), sigma=5.0)
    # plt.imshow(np.abs(test[:, 20, :]))
    # plt.show()

