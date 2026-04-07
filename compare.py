from skimage.filters import frangi as sk_frangi
from skimage.io import imread
from frangi_vesselness_filter import *
import matplotlib.pyplot as plt

# Load image and normalize 
img = imread("sample_data/01_test.tif").astype(np.float64) / 255.0

# Use green channel for retina
img = img[:, :, 1]

# Scratch implementation
v1 = frangi_filter(img, sigmas=(1, 2, 4, 8))

# skimage version
v2 = sk_frangi(img, sigmas=[1, 2, 4, 8], black_ridges=True)

# Simple threshold for retina
mask = img > 0.25 
img_masked = img * mask
v1 *= mask
v2 *= mask

# Compare
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1); plt.title("Original"); plt.imshow(img, cmap='gray')
plt.axis('off')
plt.subplot(1, 3, 2); plt.title("Custom"); plt.imshow(v1, cmap='gray')
plt.axis('off')
plt.subplot(1, 3, 3); plt.title("skimage"); plt.imshow(v2, cmap='gray')
plt.axis('off')
plt.show()