# Enhanced Image Super-Resolution using Hybrid VAE-GAN Architecture

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/matplotlib-%23007ACC.svg?style=for-the-badge&logo=matplotlib&logoColor=white)


## Introduction
Image super resolution, the process in which we improve the resolution of images to generate high-resolution counterparts, is a crucial task in computer vision related applications. In this project we have integrated the Variational Autoencoder (VAE) and Generative Adversarial Network (GAN) architectures for improving image resolution. Our method leverages the strengths of VAE in learning latent representations and GAN in generating realistic high-resolution images. 

The VAE component captures the underlying structure and distribution of high-resolution images by learning a probabilistic latent space representation, enabling efficient encoding and decoding of image features. Concurrently, the GAN component refines the generated high-resolution images by adversarially training a discriminator network to differentiate between authentic and synthesized images, thus encouraging the generator part to produce more visually pleasing results.

## Architecture

The proposed SR model architecture is a sophisticated blend of deterministic and probabilistic components, leveraging the strengths of VAEs and GANs. The model architecture comprises four main modules: pre-Encoder, Encoder, Decoder, and Discriminator.

1. Pre-Encoder: This module does preprocessing for the LR input images by applying initial feature extraction and normalization operations. Usually, it is composed of several convolutional layers, to take out important features from the LR images and reduce noise.

2. Encoder: The Encoder module encodes the pre-processed LR images into a latent space representation, capturing high-level semantic information. It typically comprises several convolutional layers with downsampling operations such as strided convolutions or max-pooling to extract hierarchical features and reduce spatial dimensions. 


3. Decoder: The Decoder module decodes the latent space representation back into HR images, generating super-resolved outputs. It is composed of transposed convolutional layers with upsampling techniques to reconstruct HR images from the learned latent space representation. Skip connections or dense connections may be employed to preserve fine details and enhance reconstruction quality. Moreover, self-attention mechanisms or attention modules can be integrated to capture long-range dependencies and improve spatial coherence.

4. Discriminator: The Discriminator module discriminates between artificially created and real HR pictures, providing adversarial training signals to the Generator (Encoder-Decoder) network. It comprises convolutional layers with binary classification output, trained to discriminate between produced and actual HR photos.


<div align="center">
  <img width="395" alt="Model Architecture" src="https://github.com/abhikalparya/ImageSuperResolution/assets/81465377/77e8a42e-ed66-4b45-b282-072e61738b5a">
</div>


## Dataset

The dataset used is sourced from publicly available image repositories and datasets such as Celeb-A, ImageNet, DIV2K, or CIFAR-10. The dataset comprises pairs of LR and HR images, where HR images serve as ground truth for training and evaluation. 

## Results

![image](https://github.com/abhikalparya/ImageSuperResolution/assets/81465377/98c94a04-69f4-4209-86f4-ad49626dc3bf)
![image](https://github.com/abhikalparya/ImageSuperResolution/assets/81465377/3debbae2-c317-4fcb-b6e6-2a43d766a688)
![image](https://github.com/abhikalparya/ImageSuperResolution/assets/81465377/070e8249-ce3f-4f4e-83a5-d572a1f7be06)
![image](https://github.com/abhikalparya/ImageSuperResolution/assets/81465377/4d2627ca-865c-4426-8056-99136c8d3364)

