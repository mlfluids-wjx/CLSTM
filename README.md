# Hybrid Conv-LSTM for nonlinear reduced-order modeling
 
A hybrid deep learning framework for nonlinear reduced-order modeling of spatiotemporal fluid dynamics, where a skip-connected convolutional autoencoder is integrated with a long short-term memory (LSTM) network.

## Introduction
- This repo provides code for the paper ["A hybrid Conv-LSTM network with skip connections for nonlinear reduced-order modeling of spatiotemporal flow fields"](under review) by Min Luo, Siqi Zhong, Jiaxin Wu and Jinlong Fu.
- This study proposes a deep-learning model that extracts fine-scale spatial features and captures long-term temporal dependencies within a low-dimensional latent space.
- A cylinder wake case is applied for illustration. 
- Please note this repository is for academical study and sharing idea, not for industrial purpose.

## Structure
    │  LICENSE
    │  README.md          
    │  main.py  (main program)
    └─ ML_CLSTM.py  (training script)
        │  ML_SkipAutoencoder.py  (network model applying skip connections)
        │  ML_NoneSkipAutoencoder.py  (network model without skip connections)
        └─ ML_utils.py  (general utilities)

## Common usage
- Run main.py

## General perparameters
- "skc (True)" for applying skip connections
- "rank" for setting subspace dimensionality     
- "offset" for setting time offset  
- ......
    
## Datasets
Obtained from Kutz, et al (http://dmdbook.com/), interpolated into 192×96 grids (Cylinder_interpolate192x96y_order_C.npy)

## Requirements
The model is built in Python environment using TensorFlow v2 backend, basically using packages of:
* Python 3.x  
* torch 
* sklearn
* numpy
