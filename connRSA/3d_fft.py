import numpy as np
from numpy.fft import fftn, ifftn
import matplotlib.pyplot as plt
from scipy import ndimage


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
    fft_data = fftn(filled_data)
    fft_shifted = np.fft.fftshift(fft_data)
    magnitude_spectrum = np.abs(fft_shifted)


    return fft_shifted, mask, magnitude_spectrum


def inverse_3d_fft_irregular(fft_shifted, mask):
    """
    Perform inverse 3D Fourier Transform and restore irregular shape

    Parameters:
    fft_data (ndarray): Fourier transformed data
    mask (ndarray): Boolean mask of valid data points

    Returns:
    ndarray: Reconstructed spatial domain data with NaNs in invalid positions
    """
    # Inverse shift
    fft_unshifted = np.fft.ifftshift(fft_shifted)

    # Perform inverse FFT
    reconstructed = ifftn(fft_unshifted)
    reconstructed = np.real(reconstructed)  # Take real part

    # Restore NaN values in invalid positions
    result = np.full_like(reconstructed, np.nan)
    result[mask] = reconstructed[mask]

    return result


def create_irregular_example(size=32):
    """Create an example of irregular 3D data"""
    data = np.full((size, size, size), np.nan)

    # Create a hollow sphere
    x, y, z = np.ogrid[-size / 2:size / 2, -size / 2:size / 2, -size / 2:size / 2]
    r = np.sqrt(x ** 2 + (y - 5) ** 2 + z ** 2)
    # shell = (r >= size / 4) & (r <= size / 3)
    shell = (r >= size / 5) & (r <= size / 3)

    data[shell] = 1.0

    # Add some random valid points
    random_points = np.random.rand(size, size, size) < 0.1
    data = np.random.normal(0.5, 0.5, size=(size, size, size))
    # data[random_points & ~shell] = 0.5

    return data


def visualize_irregular_data(original, reconstructed, spectrum, slice_idx=None):
    """Visualize original, spectrum, and reconstructed data"""
    if slice_idx is None:
        slice_idx = original.shape[0] // 2

    # for slice_idx in range(8, 24):
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.imshow(original[slice_idx], cmap='gray')
    plt.title('Original (middle slice)')
    plt.colorbar()

    plt.subplot(132)
    plt.imshow(np.log1p(spectrum[slice_idx]), cmap='gray')
    plt.title('Frequency Spectrum (log scale)')
    plt.colorbar()

    plt.subplot(133)
    plt.imshow(reconstructed[slice_idx], cmap='gray')
    plt.title('Reconstructed (middle slice)')
    plt.colorbar()

    plt.tight_layout()
    plt.show()
    quit()


# Example usage
if __name__ == "__main__":
    # Create example irregular data
    data = create_irregular_example(32)

    # Perform forward FFT
    fft_shifted, mask, magnitude_spectrum = handle_irregular_3d_fft(data)
    # print(fft_result.shape)
    # quit()

    # Perform inverse FFT
    reconstructed = inverse_3d_fft_irregular(fft_shifted, mask)

    # Calculate error only for valid points
    valid_points = ~np.isnan(data)
    error = np.abs(data[valid_points] - reconstructed[valid_points]).max()
    print(f"Maximum reconstruction error at valid points: {error}")

    # Visualize results
    visualize_irregular_data(data, reconstructed, magnitude_spectrum)