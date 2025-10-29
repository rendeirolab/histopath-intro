#!/usr/bin/env python3
"""
1b-image_analysis_basics.py

A guided introduction to basic image analysis for histopathology.

This script is part of the rotation-path-radio project and is meant to teach
foundational concepts of digital image representation and processing using Python.

Follow the comments and TODOs to complete the exercises.

Recommended environment: uv + IPython
"""

# -------------------------------------------------------------------------
# 0. Imports
# -------------------------------------------------------------------------
# Import standard scientific Python libraries for image analysis.
# (Some may need to be installed in your environment.)
# TODO: uncomment and complete as you proceed
# import numpy as np
# import imageio.v3 as iio
# import matplotlib.pyplot as plt
# from skimage import color, filters, transform

# -------------------------------------------------------------------------
# 1. Load an image into memory
# -------------------------------------------------------------------------
# The first step is to load an image file from disk.
# You can use a small histopathology image (e.g., small screenshot image from GTEx image).
# TODO: choose a local image path
image_path = "path/to/your/image.png"

# TODO: load the image into memory (hint: iio.imread)
# image = ...

# -------------------------------------------------------------------------
# 2. Explore image properties
# -------------------------------------------------------------------------
# Once loaded, inspect the following:
# - type(image)
# - image.shape  → dimensions (height, width, channels)
# - image.dtype  → pixel value type (uint8, float32, etc.)
# - value range (min, max)
#
# TODO: print out this information
# print(...)

# -------------------------------------------------------------------------
# 3. Visualize the image
# -------------------------------------------------------------------------
# Use matplotlib to display the image.
# You can adjust figure size and turn off axes for cleaner visualization.
#
# TODO: display the image with plt.imshow and plt.show()

# -------------------------------------------------------------------------
# 4. Visualize color channels
# -------------------------------------------------------------------------
# Most RGB images have three channels (Red, Green, Blue).
# Extract each channel (e.g. image[:,:,0]) and visualize them separately.
#
# TODO:
# - Create a matplotlib figure with 1 row and 3 columns of subplots
# - Show each channel in grayscale colormap
# - Add proper titles ("Red", "Green", "Blue")

# -------------------------------------------------------------------------
# 5. Perform simple image transformations
# -------------------------------------------------------------------------
# Image transformations modify geometry or orientation:
# - Resizing (scaling)
# - Rotation
# - Cropping
#
# The skimage.transform module provides convenient functions for this.
# TODO:
# - Resize the image to half its original size
# - Rotate the image by 45 degrees
# - Visualize the results side by side

# -------------------------------------------------------------------------
# 6. Apply basic segmentation and feature extraction algorithms
# -------------------------------------------------------------------------
# Simple operations from skimage.filters include:
# - Otsu thresholding (global intensity threshold)
# - Sobel edge detection (gradient-based)
#
# These are building blocks for more complex segmentation.
#
# TODO:
# - Convert the RGB image to grayscale (use color.rgb2gray)
# - Compute an Otsu threshold and create a binary mask
# - Apply Sobel filter to visualize edges
# - Plot all three: original, grayscale, edges

# -------------------------------------------------------------------------
# 7. (Optional) Build and apply custom 2D convolution kernels
# -------------------------------------------------------------------------
# Convolution allows detecting patterns such as points, lines, or circles.
# You can build small kernels manually as numpy arrays.
#
# Example:
#   kernel = np.array([[0,1,0],
#                      [1,-4,1],
#                      [0,1,0]])
#
# TODO:
# - Define 2-3 simple kernels (dot, line, donut)
# - Use scipy.signal.convolve2d or skimage.filters.rank.mean to apply them
# - Visualize responses to each kernel
#
# Observe how these filters highlight specific structures in the image.

# -------------------------------------------------------------------------
# 8. (Optional) Experiment further
# -------------------------------------------------------------------------
# Ideas for exploration:
# - Normalize image intensities to 0-1
# - Visualize histograms of pixel intensities
# - Compare results of filters on different tissue types
# - Save intermediate outputs (plt.savefig or iio.imwrite)

# -------------------------------------------------------------------------
# End of exercise
# -------------------------------------------------------------------------
print("✅ Finished image analysis basics section! Proceed to feature extraction (2).")
