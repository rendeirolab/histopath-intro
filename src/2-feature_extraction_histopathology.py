#!/usr/bin/env python3
"""
2-feature_extraction_histopathology.py

Part 2 for histopath-intro:
Feature extraction and manipulation in histopathology images.

In this exercise you will:
    - Work with whole slide images (WSI) programmatically using LazySlide
    - Segment tissue from background
    - Tile the image into smaller patches
    - Extract deep learning features from a pretrained model
    - Explore feature statistics and variability
    - Visualize the features in 2D latent spaces (PCA, UMAP)
    - Cluster tiles and visualize spatial patterns

The LazySlide package provides high-level functions that abstract most complexity,
while still allowing flexibility for experimentation.
"""

# -------------------------------------------------------------------------
# 0. Imports and setup
# -------------------------------------------------------------------------
# Import LazySlide and typical scientific Python tools
# TODO: uncomment as you go
# import lazyslide as zs
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# 1. Load a whole slide image
# -------------------------------------------------------------------------
# LazySlide provides sample data for quick testing.
# Alternatively, you can use a local .svs file (e.g., from GTEx).
#
# TODO: choose one of the following:
#   (a) use LazySlide sample data
#       wsi = zs.datasets.sample()
#   (b) load your own SVS file
#       wsi = zs.read_wsi("path/to/your_image.svs")

# -------------------------------------------------------------------------
# 2. Segment tissue from background
# -------------------------------------------------------------------------
# Tissue segmentation removes empty regions and focuses analysis
# on relevant areas.
#
# TODO: use zs.pp.find_tissues(wsi)
# Optionally, visualize segmentation results with zs.pl.tissue_mask(wsi)

# -------------------------------------------------------------------------
# 3. Tile the tissue area
# -------------------------------------------------------------------------
# Whole slide images are large; we divide them into tiles for analysis.
# A tile is a smaller image patch, often 256×256 px.
#
# TODO: run zs.pp.tile_tissues(wsi, tile_px=256, mpp=0.5)
# Then visualize the tiling results to understand spatial coverage.
#   zs.pl.tiles(wsi)

# -------------------------------------------------------------------------
# 4. Extract features with pretrained deep learning models
# -------------------------------------------------------------------------
# LazySlide supports feature extraction from various pretrained models.
# Start with a simple model (ResNet50) and later try a pathology-specific one (Virchow2).
#
# TODO:
# - Run zs.tl.feature_extraction(wsi, model="resnet50")
# - Optionally, repeat with model="virchow2"
# - Inspect the resulting feature matrix
#   features = wsi["resnet50_tiles"]

# -------------------------------------------------------------------------
# 5. Explore statistical properties of extracted features
# -------------------------------------------------------------------------
# Features are typically stored as a (n_tiles × n_features) matrix.
# You can compute per-feature mean and variance to study variability.
#
# TODO:
# - Compute mean and variance of each feature
# - Plot mean vs variance (e.g. scatter plot)
# - Identify highly variable features

# Example structure (leave implementation to student):
# means = ...
# vars_ = ...
# plt.scatter(means, vars_, s=5)
# plt.xlabel("Mean feature value")
# plt.ylabel("Variance")
# plt.title("Mean–variance relationship across features")

# -------------------------------------------------------------------------
# 6. Visualize variable features in spatial context
# -------------------------------------------------------------------------
# To see where certain features are expressed in the tissue, plot them spatially.
#
# TODO:
# - Select a few variable feature indices (e.g. top 3)
# - Visualize them using zs.pl.tiles(wsi, feature_key="resnet50", color=["1", "99"])
#   (replace "1" and "99" with your chosen indices)
#
# Observe whether distinct tissue regions exhibit different feature values.

# -------------------------------------------------------------------------
# 7. Visualize features in low-dimensional latent space
# -------------------------------------------------------------------------
# Dimensionality reduction techniques (PCA, UMAP) help visualize the
# structure of high-dimensional feature data.
#
# TODO:
# - Use zs.tl.pca(wsi, feature_key="resnet50")
# - Use zs.tl.umap(wsi, feature_key="resnet50")
# - Visualize results with zs.pl.embedding(wsi, basis="pca") and zs.pl.embedding(wsi, basis="umap")

# -------------------------------------------------------------------------
# 8. Cluster image tiles and visualize spatial clusters
# -------------------------------------------------------------------------
# Clustering groups tiles with similar morphology.
#
# TODO:
# - Run zs.tl.clustering(wsi, feature_key="resnet50", method="kmeans", n_clusters=5)
# - Visualize clusters on the WSI with zs.pl.tiles(wsi, color="cluster")
#
# Observe whether clusters correspond to distinct tissue compartments or structures.

# -------------------------------------------------------------------------
# 9. (Optional) Compare models
# -------------------------------------------------------------------------
# Repeat steps 4–8 with a foundation model like Virchow2 and compare:
# - feature distributions
# - variance explained by PCA
# - visual patterns
#
# Discuss which model captures tissue organization better.

# -------------------------------------------------------------------------
# End of exercise
# -------------------------------------------------------------------------
print("✅ Finished feature extraction and manipulation (2). Proceed to part 3!")
