# StrainNet-3D
Pytorch implementation of StrainNet-3D.

(Including training codes of SubpixelNet and RefineNet, and the affine-transformation-based 3D displacement calculation workflow)

## Introduction
- A CNN-based 3D displacement calculation method for stereo speckle images. (To realize real-time and high-precision 3D-DIC)
- Method work flow:![The workflow of StrainNet-3D method](/imgs/workflow.png)
- Simulated stereo speckle images with displacement labels (to train or to evaluate this method) can be found in my another repository [Here](https://github.com/GW-Wang-thu/Generator-of-Stereo-Speckle-images-with-displacement-labels)
- Core CNN used to calculate subpixel displacement: A light-weight CNN named SubpixelCorrNet (Architecture shows below).
![The architecture of SubpixelCorrNet](/imgs/SubpixelCorrNet.png)
- For algorithm details and the principles, please see ***(unpublished work, update later)

## Requests
- python38
- opencv-python (4.4.0 used)
- numpy (1.22.1 used)
- torch with cuda (torch1.9.0+cu111)

## Run this code
* Train the networks
    - Run ```Train_FlowNet.py```
    - If you want to train FuseNet, displacement cache file calculated by SubpixelCorrNet should be generated first.
    - If you want to generate your own training dataset, please refer to the [Stereo Speckle Generator](https://github.com/GW-Wang-thu/Generator-of-Stereo-Speckle-images-with-displacement-labels)
* 2-D displacement (and deformation) calculation
    - Run ```DispCalculator2D.py```, remember to shift the model filepath and the image filepath to your own ones.
    - Strain calculation using gradient filter technique is provided.
* 3-D displacement (and deformation) calculation
    - Run ```DispCalculator3D.py```, remember to shift the model filepath and the image filepath to your own ones.
    - The calibration parameters and other settings should be set correctly.
    - Simple strain calculation code using gradient filter is provided.

## Pre-trained model
- Pre-trained parameter file of SubpixelCorrNet can be download from [Google drive](https://drive.google.com/drive/folders/17fP3m60Ab5OKycFhSUtXHN4j7IToi5Np?usp=sharing).

## Demonstration
- Comparison of the 3D displacement calculated using StrainNet3D and 3D-DIC of one test set.
![Comparison of 3D displacement calculated using StrainNet-3D and 3D-DIC](/imgs/uvw_comparison.png)
<center>Tabel.1 Comparison of Mean Absolute Error(MAE) of the results (Pixels)</center>

![Tabel1](/imgs/precision_comparison.png) 
<center>Tabel.2 Comparison of the calculation speed(POI/s) </center>

![Tabel2](/imgs/precision_comparison.png)

- Experimental speckle images calculation in extreme light conditions.
![experiment calculation](/imgs/experiment_calculation.png)

- Light-changing real-time displacement monitoring demo.
The Realtime demo video can be found [here](https://drive.google.com/drive/folders/17fP3m60Ab5OKycFhSUtXHN4j7IToi5Np?usp=sharing)
![RealtimeDemo](/imgs/monitoring.png)



## Cite this work

