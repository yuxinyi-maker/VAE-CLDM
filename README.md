# VAE-CLDM: Condition-Guided Latent Diffusion for Efficient 3D Digital Rock Reconstruction

This repository contains the implementation of **VAE-CLDM**, a two-stage conditional latent diffusion framework for condition-guided reconstruction of 3D porous geomaterials. The model uses a 3D VAE to compress voxelized porous structures into a compact latent space and trains a conditional latent diffusion model using pore-scale descriptors as control variables.

## Main features

* 3D VAE-based latent-space compression of voxelized porous structures.
* Conditional latent diffusion for condition-guided 3D reconstruction.
* FiLM-style condition injection using porosity, mean pore diameter, and pore-size standard deviation.
* Scripts for VAE pre-training, CLDM training, conditional generation.

## Installation

A conda environment is recommended:

```bash
conda create -n vae-cldm python=3.8 -y
conda activate vae-cldm
```

Install PyTorch according to your CUDA version, then install the main dependencies:

```bash
pip install numpy scipy scikit-image scikit-learn matplotlib pandas tqdm einops pytorch-lightning tensorboard
```

Optional packages for visualization and porous-media analysis:

```bash
pip install pyvista porespy
```

## Basic usage

1. Prepare binary `.npy` voxel samples with shape `(64, 64, 64)`.
2. Pre-train the VAE:

```bash
python train\_vae.py
```

3. Train the CLDM in the fixed VAE latent space:

```bash
python main\_vaedpm.py
```

4. Generate condition-guided reconstructions:

```bash
python generate.py
```



Before running these scripts, update dataset paths, checkpoint paths, output directories, and GPU settings according to your local environment.

## Data format

Input data should be binary 3D NumPy arrays: (64, 64, 64)

For pore-size conditions, filenames should follow: <pore\_size\_mean>\_<pore\_size\_std>.npy

Porosity is computed from the voxel data during loading, while mean pore diameter and pore-size standard deviation are parsed from filenames.

## Computational requirements

The experiments in the manuscript were performed using an NVIDIA Tesla V100 GPU with 32 GB VRAM. For training on `64^3` voxel samples, a CUDA-enabled GPU with at least 16 GB VRAM is recommended; 24 GB or more is preferred.

CPU-only execution is not recommended for training, but may be sufficient for small debugging tasks or parameter counting.

## Documentation

Detailed instructions are provided in: user\_manual.txt

## Contact

Xinyi Yu, 18407870306

