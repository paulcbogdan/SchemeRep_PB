import numpy as np
from numpy.fft import fftn, ifftn
import matplotlib.pyplot as plt


def create_shifted_ring(size=32, radius=8, thickness=2, shift=(5, 3, 0)):
    """Create a 3D ring with a spatial shift"""
    data = np.zeros((size, size, size))
    x, y, z = np.ogrid[-size / 2:size / 2, -size / 2:size / 2, -size / 2:size / 2]
    # Apply spatial shift
    x = x + shift[0]
    y = y + shift[1]
    z = z + shift[2]
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    ring = (r >= radius - thickness / 2) & (r <= radius + thickness / 2)
    data[ring] = 1.0
    return data


def analyze_phase_and_magnitude(data1, data2):
    """Analyze and compare FFTs of two datasets"""
    # Compute FFTs
    fft1 = fftn(data1)
    fft2 = fftn(data2)

    # Shift FFTs to center
    fft1_shifted = np.fft.fftshift(fft1)
    fft2_shifted = np.fft.fftshift(fft2)

    # Extract magnitude and phase
    magnitude1 = np.abs(fft1_shifted)
    magnitude2 = np.abs(fft2_shifted)
    phase1 = np.angle(fft1_shifted)
    phase2 = np.angle(fft2_shifted)

    # Phase difference
    phase_diff = np.angle(fft2_shifted * np.conj(fft1_shifted))

    return magnitude1, magnitude2, phase1, phase2, phase_diff


def visualize_phase_analysis(data1, data2, mag1, mag2, phase1, phase2, phase_diff):
    """Visualize the spatial and frequency domain data"""
    mid_slice = data1.shape[0] // 2

    plt.figure(figsize=(20, 10))

    # Original space domain
    plt.subplot(251)
    plt.imshow(data1[mid_slice], cmap='gray')
    plt.title('Ring 1 (Space Domain)')
    plt.colorbar()

    plt.subplot(256)
    plt.imshow(data2[mid_slice], cmap='gray')
    plt.title('Ring 2 (Space Domain)')
    plt.colorbar()

    # Magnitude spectra
    plt.subplot(252)
    plt.imshow(np.log1p(mag1[mid_slice]), cmap='viridis')
    plt.title('Magnitude Spectrum 1')
    plt.colorbar()

    plt.subplot(257)
    plt.imshow(np.log1p(mag2[mid_slice]), cmap='viridis')
    plt.title('Magnitude Spectrum 2')
    plt.colorbar()

    # Phase information
    plt.subplot(253)
    plt.imshow(phase1[mid_slice], cmap='hsv')
    plt.title('Phase Spectrum 1')
    plt.colorbar()

    plt.subplot(258)
    plt.imshow(phase2[mid_slice], cmap='hsv')
    plt.title('Phase Spectrum 2')
    plt.colorbar()

    # Phase difference
    plt.subplot(254)
    plt.imshow(phase_diff[mid_slice], cmap='hsv')
    plt.title('Phase Difference')
    plt.colorbar()

    # Reconstruction test
    # Take magnitude from first and phase from second
    mixed_fft = np.fft.ifftshift(mag1 * np.exp(1j * phase2))
    mixed_space = np.real(ifftn(mixed_fft))

    plt.subplot(259)
    plt.imshow(mixed_space[mid_slice], cmap='gray')
    plt.title('Mag1 × Phase2 Reconstruction')
    plt.colorbar()

    plt.tight_layout()
    plt.show()


# Create two rings with different positions
ring1 = create_shifted_ring(shift=(0, 0, 0))
ring2 = create_shifted_ring(shift=(5, 3, 0))

# Analyze and visualize
mag1, mag2, phase1, phase2, phase_diff = analyze_phase_and_magnitude(ring1, ring2)
visualize_phase_analysis(ring1, ring2, mag1, mag2, phase1, phase2, phase_diff)

# Demonstrate phase-magnitude relationship with a simple 1D example
x = np.linspace(0, 10, 100)
signal1 = np.sin(2 * np.pi * x)
signal2 = np.sin(2 * np.pi * x + np.pi / 2)  # 90-degree phase shift

plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.plot(x, signal1, label='Original')
plt.plot(x, signal2, label='Phase Shifted')
plt.title('1D Signals')
plt.legend()

fft1 = np.fft.fft(signal1)
fft2 = np.fft.fft(signal2)

plt.subplot(132)
plt.plot(np.abs(fft1), label='|FFT1|')
plt.plot(np.abs(fft2), label='|FFT2|')
plt.title('Magnitude Spectra')
plt.legend()

plt.subplot(133)
plt.plot(np.angle(fft1), label='Phase 1')
plt.plot(np.angle(fft2), label='Phase 2')
plt.title('Phase Spectra')
plt.legend()

plt.tight_layout()
plt.show()