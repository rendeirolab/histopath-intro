# histopath-intro

<a href='https://codespaces.new/rendeirolab/histopath-intro'><img src='https://github.com/codespaces/badge.svg' alt='Open in GitHub Codespaces' style='max-width: 100%;'></a>

A small project as an entry point to working with histopathology data from a deep learning perspective.

This repository was created with a [cookiecutter template](https://github.com/rendeirolab/_project_template), version 0.4.1.

## Project goals
- To hone skills in scientific programming with Python
- To learn about histopathology and its analysis with deep learning
- To utilize deep learning features to extract quantitative information from histopathology images

## Analysis/tasks

### 0. Get familiar with the data and set up a working environment
- [ ] Set up a text editor/IDE. [VScode](https://code.visualstudio.com/) or [SublimeText](https://www.sublimetext.com/) recommended.
- [ ] [optional] Try out Github codespaces as a working environment (see badge on top).
- [ ] [optional] Set up [Obsidian](https://obsidian.md/) for note taking.


### 1. Basic image processing with Python

- [ ] Understand how images are represented in the computer and their attributes (shape, color, bit depth)
- [ ] Use `numpy`, `imageio`, `scikit-image`, and `matplotlib` to load, manipulate, and visualize images:
    - [ ] Load one image into memory, check its dimensions (shape) and bit depth
    - [ ] Visualize it with matplotlib
    - [ ] Visualize each channel independently, with all plotted as subplots in the same figure
    - [ ] Transform image (resize, rotate, )
    - [ ] Explore basic algorithms for image segmentation and feature extraction (otsu, sobel, etc)
    - [ ] Build a set of 2D basic kernels by hand (e.g. points, circles/donuts, lines) and convolute them across the image to extract features


### 2. Histopathology analysis basics

#### 2a. Data exploration

- [ ] Go to https://gtexportal.org/home/histologyPage, and see images from various tissues together with their annotations, download some locally
- [ ] [optional] Download [QuPath](https://qupath.github.io/) and use it locally to see one images from the GTEx project

#### 2b. Feature extraction and manipulation in histopathology

- [ ] Download one whole slide image (SVS file)
- [ ] Use [LazySlide](https://github.com/rendeirolab/LazySlide) package to work with it programatically - [docs here](https://lazyslide.readthedocs.io/en/stable/):
    - [ ] Segment the tissue area from background and visualize it
    - [ ] Tile the whole slide image to produce smaller images (e.g. 256x256 pixels) and visualize them
- [ ] Extract features from the image using a pretrained deep learning model. Use a basic vision model e.g. ResNet50 and a foundation model e.g. Virchow2
- [ ] Observe the statistical distribution of the features (mean vs variance relationship)
- [ ] Select variable features and visualize them in their original spatial conformation
- [ ] Visualize features using PCA and UMAP
- [ ] Cluster image tiles and visualize clusters spatially

### 3. Analysis of tissues based on deep morphological features

- [ ] Use a representation of all slides in GTEx based on a variety of foundation models for pathology
- [ ] Visualize latent spaces derived from each model
- [ ] Investigate the variance explained by each model in light of donor variables (e.g. age, sex, cohort)
- [ ] Detect features most specific to each tissue/organ and visualize those for each slide in a clustered heatmap with annotations for the slides
- [ ] Build a predictor of age from these models
    - [ ] Use the morphological features as predictors, age as the target variable
    - [ ] Choose a classical machine learning model for regression (linear regression, SVM, random forest regressor), use cross validation, and choose appropriate metrics
    - [ ] Visualize the performance metrics and the results by contrasting the known and predicted ages directly

## Organization

- Raw data  is under the `data` directory (likely empty in a remote repository on GitHub).
- The [src](src) directory contains source code used to analyze the data.
- The [metadata](metadata) directory can hold metadata relevant to annotate the samples.
- Outputs from the analysis are present in the `results` directory, with subfolders pertaining to each part of the analysis as described below.

## Practical instructions

This small project should be able to run in any machine with Python 3.8+ installed, with at least 8GB of RAM and a decent CPU. A GPU is not strictly necessary, but will speed up feature extraction from deep learning models significantly.

If you have access to a server or HPC cluster, that is also a good option.

### Environment

We use [`uv`](https://github.com/astral-sh/uv) to manage software dependencies and development environments.
Follow the instructions to install `uv` in any machine:

- Linux and MacOS: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Windows: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`


### Runnning code

First make sure to clone this repository locally:
```
git clone git@github.com:rendeirolab/histopath-intro.git
```
SSH is the recommended way, but requires a little more configuration. [Read more here](https://docs.github.com/en/authentication/connecting-to-github-with-ssh).

Then in the cloned repository, create a virtual environment in the root of the project: `uv sync`. This will read in the [`pyproject.toml`](pyproject.toml) file which specifies the software dependencies, and use it.

To add new software dependencies use `uv add <dependency name>`. `uv sync` will be run automatically after adding a new dependency.

I recomment using IPython to work interactively. Use a text editor/IDE of your choice to edit the source code and run it side-by-side in IPython, line-by-line during development. Run the following command to start IPython:

```bash
uv run --with IPython ipython
```

Jupyter notebooks are also okay to use. Just be sure to use 

After development, to run a whole script, you may run a full script with the following:
```bash
uv run src/analysis_script.py  # replace analysis_script.py with the file you want to run
```

Be sure to [check out the full `uv` docs](https://docs.astral.sh/uv/) to learn about the amazing tool.
