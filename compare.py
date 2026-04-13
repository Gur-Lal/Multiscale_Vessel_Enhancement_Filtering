import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.filters import frangi as sk_frangi, meijering, sato
from frangi_vesselness_filter import *

# Load image and normalize 
img = imread("sample_data/01_test.tif").astype(np.float64) / 255.0

# Use green channel for retina
img = img[:, :, 1]

# Scratch implementation
v_custom = frangi_filter(img, sigmas=(1, 2, 4, 8))

# skimage versions
v_frangi = sk_frangi(img, sigmas=[1, 2, 4, 8], black_ridges=True)
v_meijering = meijering(img, sigmas=[0.5, 1, 2, 4], black_ridges=True)
v_sato = sato(img, sigmas=[0.5, 1, 2, 4], black_ridges=True)

# Simple threshold for retina
mask = img > 0.25
img_masked = img * mask

v_custom *= mask
v_frangi *= mask
v_meijering *= mask
v_sato *= mask

# Plot comparison
plt.figure(figsize=(8, 3))
plt.subplot(2, 3, 1); plt.title("Original"); plt.imshow(img, cmap='gray'); plt.axis('off')
plt.subplot(2, 3, 2); plt.title("Custom Frangi"); plt.imshow(v_custom, cmap='gray'); plt.axis('off')
plt.subplot(2, 3, 3); plt.title("skimage Frangi"); plt.imshow(v_frangi, cmap='gray'); plt.axis('off')
plt.subplot(2, 3, 4); plt.title("Meijering"); plt.imshow(v_meijering, cmap='gray'); plt.axis('off')
plt.subplot(2, 3, 5); plt.title("Sato"); plt.imshow(v_sato, cmap='gray'); plt.axis('off')
plt.show()