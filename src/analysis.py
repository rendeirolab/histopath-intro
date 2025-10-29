#!/usr/bin/env python3

"""
This module imports various libraries and modules required for the project.
Useful if you want to run an interactive session with all dependencies loaded.
"""


import typing as tp
import zipfile
from functools import partial
from pathlib import Path
import glob
import os

from scipy.stats import f_oneway
from skimage import color, filters, transform
from skimage import filters
from skimage import transform, filters, measure, exposure
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from umap import UMAP
import imageio.v3 as iio
import lazyslide as zs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import torch
import torchvision.models as models
import torchvision.transforms as T

print("✅ All modules imported successfully.")
