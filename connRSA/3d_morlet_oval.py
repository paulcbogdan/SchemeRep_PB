import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize_3d_morlet(size=32, wavelength=8, direction=(1, 0, 0), sigma=1.0):
    """
    Create and visualize a 3D Morlet wavelet using multiple plotting methods.

    Parameters:
    -----------
    size : int
        Size of the wavelet grid
    wavelength : float
        Wavelength of oscillations
    direction : tuple
        Direction of wavelet oscillation
    sigma : float
        Width of Gaussian envelope
    """
    # Create coordinate grids
    x = np.linspace(-2, 2, size)
    y = np.linspace(-2, 2, size)
    z = np.linspace(-2, 2, size)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Normalize direction vector
    direction = np.array(direction)
    direction = direction / np.linalg.norm(direction)

    # Calculate projection and perpendicular distance
    projection = (X * direction[0] + Y * direction[1] + Z * direction[2])
    perpendicular_dist = np.sqrt(X ** 2 + Y ** 2 + Z ** 2 - projection ** 2)

    # Create wavelet
    k = 2 * np.pi / wavelength
    gaussian = np.exp(-(perpendicular_dist ** 2 + projection ** 2) / (2 * sigma ** 2))
    oscillation = np.exp(1j * k * projection)
    wavelet = gaussian * oscillation

    # print(wavelet.shape)
    # quit()


    # Create figure with multiple subplots
    fig = plt.figure(figsize=(15, 5))

    # Plot 1: XY slice at middle Z
    ax1 = fig.add_subplot(131)
    mid_z = size // 2
    im1 = ax1.imshow(np.real(wavelet[:, :, mid_z]).T,
                     extent=[-2, 2, -2, 2],
                     cmap='RdBu')
    ax1.set_title('XY Slice (Real Part)')
    plt.colorbar(im1, ax=ax1)

    # Plot 2: XZ slice at middle Y
    ax2 = fig.add_subplot(132)
    mid_y = size // 2
    im2 = ax2.imshow(np.real(wavelet[:, mid_y, :]).T,
                     extent=[-2, 2, -2, 2],
                     cmap='RdBu')
    ax2.set_title('XZ Slice (Real Part)')
    plt.colorbar(im2, ax=ax2)

    # Plot 3: 3D isosurface
    ax3 = fig.add_subplot(133, projection='3d')

    # Create isosurface using contour
    magnitude = np.abs(wavelet)
    max_val = np.max(magnitude)
    threshold = max_val * 0.3  # Show isosurface at 30% of max value

    X_grid, Y_grid, Z_grid = np.meshgrid(x, y, z, indexing='ij')

    # Plot negative and positive isosurfaces separately with correct level ordering
    ax3.contour(X_grid[:, :, size // 2],
                Y_grid[:, :, size // 2],
                np.real(wavelet[:, :, size // 2]),
                levels=[-threshold],
                colors=['blue'])

    ax3.contour(X_grid[:, :, size // 2],
                Y_grid[:, :, size // 2],
                np.real(wavelet[:, :, size // 2]),
                levels=[threshold],
                colors=['red'])

    ax3.set_title('3D Isosurface View')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')

    plt.tight_layout()
    return fig


# Create visualizations with different parameters
def show_wavelet_examples():
    # Example 1: Basic wavelet along x-axis
    fig1 = visualize_3d_morlet(size=32, wavelength=8, direction=(1, 0, 0), sigma=1.0)
    fig1.suptitle('Basic Morlet Wavelet (X-direction)')

    # Example 2: Wavelet along diagonal direction
    fig2 = visualize_3d_morlet(size=32, wavelength=8,
                               direction=(0, 0, 0), sigma=1.0)
    fig2.suptitle('Diagonal Morlet Wavelet')

    # Example 3: Wavelet with different wavelength
    fig3 = visualize_3d_morlet(size=32, wavelength=2,
                               direction=(1, 0, 0), sigma=1.0)
    fig3.suptitle('Higher Frequency Wavelet')

    plt.show()


# Run the visualization
show_wavelet_examples()