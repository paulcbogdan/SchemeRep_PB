import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def normalize_fft_frequencies(data, physical_dimensions, target_resolution=None, preserve_highest_freq=True):
    """
    Normalize FFT frequencies for non-cubic domains with proper frequency shifting.

    Parameters:
    data: np.ndarray
        The 3D FFT result array
    physical_dimensions: tuple
        Physical dimensions (Lx, Ly, Lz) of the domain
    target_resolution: int, optional
        Desired number of points for uniform frequency grid
    preserve_highest_freq: bool
        If True, preserves highest frequencies across all dimensions

    Returns:
    tuple:
        - Frequency grids (kx, ky, kz)
        - Interpolated FFT data
        - Magnitude of wave vectors
    """
    Nx, Ny, Nz = data.shape
    Lx, Ly, Lz = physical_dimensions

    # Calculate frequency spacing for each dimension
    dx = Lx / Nx
    dy = Ly / Ny
    dz = Lz / Nz

    # Create original frequency arrays and shift them for interpolation
    kx_orig = np.fft.fftfreq(Nx, dx)
    ky_orig = np.fft.fftfreq(Ny, dy)
    kz_orig = np.fft.fftfreq(Nz, dz)

    # Convert to physical frequencies
    kx_orig = 2 * np.pi * kx_orig
    ky_orig = 2 * np.pi * ky_orig
    kz_orig = 2 * np.pi * kz_orig

    # Shift frequencies and data for interpolation
    kx_orig = np.fft.fftshift(kx_orig)
    ky_orig = np.fft.fftshift(ky_orig)
    kz_orig = np.fft.fftshift(kz_orig)
    data_shifted = np.fft.fftshift(data)

    # Calculate frequency parameters
    dk_x = abs(kx_orig[1] - kx_orig[0])
    dk_y = abs(ky_orig[1] - ky_orig[0])
    dk_z = abs(kz_orig[1] - kz_orig[0])

    k_max_x = abs(kx_orig).max()
    k_max_y = abs(ky_orig).max()
    k_max_z = abs(kz_orig).max()

    if preserve_highest_freq:
        dk_target = min(dk_x, dk_y, dk_z)
        k_max_target = max(k_max_x, k_max_y, k_max_z)
        required_resolution = int(np.ceil(2 * k_max_target / dk_target))
        required_resolution += required_resolution % 2

        if target_resolution is not None and target_resolution < required_resolution:
            print(f"Warning: target_resolution {target_resolution} is too small to preserve all frequencies.")
            print(f"Need at least {required_resolution} points. Using required resolution instead.")

        target_resolution = required_resolution if target_resolution is None else max(target_resolution,
                                                                                      required_resolution)
    else:
        dk_target = max(dk_x, dk_y, dk_z)
        k_max_target = min(k_max_x, k_max_y, k_max_z)
        target_resolution = target_resolution if target_resolution is not None else max(Nx, Ny, Nz)

    # Create uniform frequency grid
    k_uniform = np.linspace(-k_max_target, k_max_target, target_resolution)

    # Create interpolators for shifted data
    interpolator_real = RegularGridInterpolator(
        (kx_orig, ky_orig, kz_orig),
        data_shifted.real,
        bounds_error=False,
        fill_value=0
    )

    interpolator_imag = RegularGridInterpolator(
        (kx_orig, ky_orig, kz_orig),
        data_shifted.imag,
        bounds_error=False,
        fill_value=0
    )

    # Create uniform 3D grid
    KX, KY, KZ = np.meshgrid(k_uniform, k_uniform, k_uniform, indexing='ij')
    points = np.column_stack((KX.ravel(), KY.ravel(), KZ.ravel()))

    # Interpolate
    data_uniform_real = interpolator_real(points).reshape(KX.shape)
    data_uniform_imag = interpolator_imag(points).reshape(KX.shape)
    data_uniform = data_uniform_real + 1j * data_uniform_imag

    # Shift back
    data_uniform = np.fft.ifftshift(data_uniform)
    k_uniform = np.fft.ifftshift(k_uniform)

    # Calculate wave vector magnitude
    K = np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)

    return k_uniform, data_uniform, K


def visualize_frequency_grids(original_dims, physical_dims, target_resolution=None, preserve_highest_freq=True):
    """
    Visualize the original and normalized frequency grids.
    """
    Nx, Ny, Nz = original_dims
    Lx, Ly, Lz = physical_dims

    # Create dummy data
    dummy_data = np.random.random(original_dims) + 1j * np.random.random(original_dims)

    # Get normalized frequencies
    k_uniform, _, _ = normalize_fft_frequencies(
        dummy_data,
        physical_dims,
        target_resolution,
        preserve_highest_freq
    )

    # Create original frequency grids
    kx_orig = 2 * np.pi * np.fft.fftfreq(Nx, Lx / Nx)
    ky_orig = 2 * np.pi * np.fft.fftfreq(Ny, Ly / Ny)
    kz_orig = 2 * np.pi * np.fft.fftfreq(Nz, Lz / Nz)

    # Create figure
    fig = plt.figure(figsize=(15, 5))

    # Plot original frequencies
    ax1 = fig.add_subplot(121, projection='3d')
    KX, KY, KZ = np.meshgrid(kx_orig, ky_orig, kz_orig, indexing='ij')
    ax1.scatter(KX.flatten(), KY.flatten(), KZ.flatten(),
                c='blue', alpha=0.6, label='Original')
    ax1.set_title('Original Frequency Grid')

    # Plot normalized frequencies
    ax2 = fig.add_subplot(122, projection='3d')
    KX, KY, KZ = np.meshgrid(k_uniform, k_uniform, k_uniform, indexing='ij')
    ax2.scatter(KX.flatten(), KY.flatten(), KZ.flatten(),
                c='red', alpha=0.6, label='Normalized')
    ax2.set_title('Normalized Frequency Grid')

    # Labels and formatting
    for ax in [ax1, ax2]:
        ax.set_xlabel('kx')
        ax.set_ylabel('ky')
        ax.set_zlabel('kz')
        ax.legend()

    plt.tight_layout()
    return fig


def plot_frequency_spectrum(data, k_values, axis=0, log_scale=True):
    """
    Plot the frequency spectrum along a specified axis.
    """
    # Take slice along specified axis
    if axis == 0:
        spectrum = np.abs(data[:, data.shape[1] // 2, data.shape[2] // 2])
    elif axis == 1:
        spectrum = np.abs(data[data.shape[0] // 2, :, data.shape[2] // 2])
    else:
        spectrum = np.abs(data[data.shape[0] // 2, data.shape[1] // 2, :])

    plt.figure(figsize=(10, 5))
    if log_scale:
        plt.semilogy(k_values, spectrum)
    else:
        plt.plot(k_values, spectrum)

    plt.grid(True)
    plt.xlabel(f'k{"xyz"[axis]}')
    plt.ylabel('Magnitude')
    plt.title(f'Frequency Spectrum Along {"XYZ"[axis]}-axis')
    return plt.gcf()


# Example usage
def example_with_visualizations():
    # Create sample dimensions
    Nx, Ny, Nz = 32, 40, 24
    Lx, Ly, Lz = 1.0, 1.25, 0.75

    # Create sample data (example: multiple wave patterns)
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    z = np.linspace(0, Lz, Nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Create sample wave pattern with different frequencies
    wave = (np.sin(2 * np.pi * X / Lx) * np.cos(2 * np.pi * Y / Ly) * np.exp(-Z / Lz) +
            np.sin(6 * np.pi * X / Lx) * np.cos(4 * np.pi * Y / Ly) * np.exp(-2 * Z / Lz))

    wave = np.random.normal(0.5, 0.5, size=(Nx, Ny, Nz))

    # Compute FFT
    fft_data = np.fft.fftn(wave)

    # Normalize frequencies
    k_uniform, data_uniform, K = normalize_fft_frequencies(
        fft_data,
        (Lx, Ly, Lz),
        preserve_highest_freq=True
    )

    plt.imshow(np.abs(data_uniform[16]), cmap='gray')
    plt.show()
    quit()

    # Visualize grids
    grid_fig = visualize_frequency_grids(
        (Nx, Ny, Nz),
        (Lx, Ly, Lz),
        preserve_highest_freq=True
    )



    # Plot spectrum along each axis
    # spectrum_figs = []
    # for axis in range(3):
    #     fig = plot_frequency_spectrum(data_uniform, k_uniform, axis=axis)
    #     spectrum_figs.append(fig)
    #
    # return grid_fig, spectrum_figs

if __name__ == "__main__":
    example_with_visualizations()
    # plt.show()
    # quit()