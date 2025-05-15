# Comparison of Latent Space Interpretability
We investigate the semantic interpretability of latent spaces for several generative models including StyleGAN2, Variational Autoencoders (VAEs), $\beta$-VAEs, and $\beta$-TCVAEs. Using the CelebA and CelebA-HQ datasets, with the "Eyeglasses" attribute as a test case, we explore semantic manipulation in latent spaces by training linear SVM classifiers and analyzing their effect on generated outputs. We compare results both quantitatively and qualitatively, and discuss which models yield more interpretable latent representations.

This repository contains the code used for our CSCI5527 Deep Learning Project at the University of Minnesota, Twin Cities. The project was completed by:
Isaac Berlin,
Tianzhe Han,
Alex Iliarski,
Josh Krueger,

## How to Run the Code
- clone this repository
- download the CelebA dataset from [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and update file paths in the code

### StyleGAN2 Specific
- clone the StyleGAN2-ada-pytorch repository from [here](https://github.com/NVlabs/stylegan2-ada-pytorch)
- download the pretrained model from [here](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/research/models/stylegan2) and update file paths in the code
- install visual studio if on windows and ensure you have the correct version of CUDA installed

### VAE Specific
- clone the VAE-pytorch repository from [here](https://github.com/AntixK/PyTorch-VAE/tree/master)

