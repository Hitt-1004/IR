import cv2
from skimage.feature import local_binary_pattern
import numpy as np
import matplotlib.pyplot as plt

# Load an image using OpenCV
image = cv2.imread('image.png')

# Calculate the color histograms
hist_red = cv2.calcHist([image], [0], None, [256], [0, 256])
hist_green = cv2.calcHist([image], [1], None, [256], [0, 256])
hist_blue = cv2.calcHist([image], [2], None, [256], [0, 256])

# Convert the image to grayscale for LBP texture features
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Calculate LBP texture features
radius = 1
n_points = 8 * radius
lbp_image = local_binary_pattern(image_gray, n_points, radius, method='uniform')

# Extract the unique LBP patterns and their frequencies
lbp_hist, _ = np.histogram(lbp_image, bins=np.arange(0, n_points + 3), range=(0, n_points + 2))

# Print and visualize the results
plt.figure(figsize=(12, 6))

# Plot the color histograms
plt.subplot(1, 2, 1)
plt.plot(hist_red, color='red')
plt.plot(hist_green, color='green')
plt.plot(hist_blue, color='blue')
plt.title('Color Histograms')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

# Plot the LBP texture features
plt.subplot(1, 2, 2)
plt.plot(lbp_hist, color='black')
plt.title('LBP Texture Features')
plt.xlabel('LBP Pattern')
plt.ylabel('Frequency')
plt.show()
