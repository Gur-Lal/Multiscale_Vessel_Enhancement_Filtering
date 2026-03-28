import numpy as np
from scipy.ndimage import gaussian_filter

def hessian2d(image, sigma):
    """Compute 2D Hessian using Gaussian derivatives."""
    I = image.astype(np.float64)

    Dxx = gaussian_filter(I, sigma=sigma, order=(2, 0))
    Dxy = gaussian_filter(I, sigma=sigma, order=(1, 1))
    Dyy = gaussian_filter(I, sigma=sigma, order=(0, 2))

    return Dxx, Dxy, Dyy

def eigvals2d(Dxx, Dxy, Dyy):
    """Compute eigenvalues of 2x2 Hessian matrix."""
    tmp = np.sqrt((Dxx - Dyy)**2 + 4 * Dxy**2)

    lambda1 = 0.5 * (Dxx + Dyy - tmp)
    lambda2 = 0.5 * (Dxx + Dyy + tmp)

    # sort so |lambda1| <= |lambda2|
    swap = np.abs(lambda1) > np.abs(lambda2)
    lambda1[swap], lambda2[swap] = lambda2[swap], lambda1[swap]

    return lambda1, lambda2

def frangi_filter(image,sigmas=(1, 2, 4, 8),beta=0.5,c=0.02,black_ridges=True):
    # Denoise
    image = gaussian_filter(image, sigma=1)
    """Minimal Frangi vesselness filter (2D)."""
    image = image.astype(np.float64)

    vesselness = np.zeros_like(image)

    for sigma in sigmas:
        # Hessian
        Dxx, Dxy, Dyy = hessian2d(image, sigma)

        # Scale normalization
        Dxx *= sigma**2
        Dxy *= sigma**2
        Dyy *= sigma**2

        # Eigenvalues
        lambda1, lambda2 = eigvals2d(Dxx, Dxy, Dyy)

        # Avoid division by zero
        eps = np.finfo(float).eps
        lambda2 = np.where(np.abs(lambda2) < eps, eps, lambda2)

        # Measures
        Rb = (lambda1 / lambda2) ** 2
        S2 = lambda1**2 + lambda2**2

        # Frangi vesselness
        V = np.exp(-Rb / (2 * beta**2)) * (1 - np.exp(-S2 / (2 * c**2)))

        # Ridge polarity
        if black_ridges:
            V[lambda2 < 0] = 0
        else:
            V[lambda2 > 0] = 0

        V = V / (sigma**2)
        V[V < 0.1 * V.max()] = 0
        vesselness = np.maximum(vesselness, V)

    vesselness = (vesselness - vesselness.min()) / (vesselness.max() - vesselness.min())
    return vesselness