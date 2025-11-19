import numpy as np

def fourier_descriptor(pts, n=32):
    # Convert 2D points into complex numbers
    p = pts[:, 0] + 1j * pts[:, 1]
    
    # Remove mean (translation invariance)
    p = p - p.mean()
    
    # Compute FFT and keep first n coefficients
    fd = np.fft.fft(p)[:n]
    
    # Normalize by the magnitude of the first non-DC component
    if abs(fd[1]) != 0:
        fd = fd / abs(fd[1])
    
    # Return magnitudes (rotation invariance)
    return np.abs(fd)
