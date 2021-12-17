# Semantic Segmentation using Deep Neural Networks for MAVs


In this work, we evaluate the performance of state-of-the-art methods such as Recurrent Neural Networks (RNNs), 3D Convolutional Neural Networks (CNNs), and optical flow for video semantic segmentation in terms of accuracy and inference speed on three datasets with different camera motion configurations. The results show that using an RNN with convolutional operators outperforms all methods and achieves a performance boost of 10.8\% on the KITTI (MOTS) dataset with 3 degrees of freedom (DoF) motion and a small 0.6\% improvement on the CyberZoo dataset with 6 DoF motion over the frame-based semantic segmentation method. The inference speed was measured on the CyberZoo dataset, achieving 321 fps on an NVIDIA GeForce RTX 2060 GPU and 30 fps on a NVIDIA Jetson TX2. 

![alt text](https://github.com/tommyvtran97/MAV-Segmentation/blob/master/Media/MAVRNN.png)

## Installation
Create a conda environment with Python 3.8:

```
conda create -n python_env python=3.8
```

Install the following python packages:

* numpy
* matplotlib
* pytorch
* opencv
* cupy (version should match with cuda, for example cupy 10.1 with cuda 10.1, only required to run LiteFlowNet)

Copy the content in the dataset folder as follows:

```
├── Dataset
│   └── CyberZoo/*                        # Folder containing CyberZoo dataset
│   └── KITTI_Cars/*                      # Folder containing KITTI (MOTS) dataset
│   └── KITTI_Pedestrians/*               # Folder containing KITTI (MOTS Challenge) dataset
```

## Usage 
The content consists of 5 networks; RGBF, RGBF_Fusion, U-Net, U-Net_3D, and U-Net_RNN. We show an example for the U-Net_RNN folder, the same instructions hold for the other folders. The training settings can be changed in the `RNN_train.py`. To train the network run:

```
cd U-Net_RNN
python RNN_train.py
```

To show the results of U-Net, RNN and LSTM. Use '1' for the KITTI dataset and '2' for the CyberZoo dataset:

```
python results.py --dataset 1
```

The results below show from top to bottom: RGB - GT - U-Net - RNN - LSTM

![alt text](https://github.com/tommyvtran97/MAV-Segmentation/blob/master/Media/results.png)

## Content and Folder Structure
In this section, a detailed overview of the structure of the folder is presented.

```
├── Dataset
│   └── CyberZoo/*                        # Folder containing CyberZoo dataset
│   └── KITTI_Cars/*                      # Folder containing KITTI (MOTS) dataset
│   └── KITTI_Pedestrians/*               # Folder containing KITTI (MOTS Challenge) dataset
├── Figures
│   └── Examples
│       └── CyberZoo/*                    # Folder containing results on CyberZoo dataset
│       └── KITTI_Cars/*                  # Folder containing results on KITTI (MOTS) dataset
│       └── KITTI_Pedestrians/*           # Folder containing results on KITTI (MOTS Challenge) dataset
│   └── Results
│       └── Cars_Inference_IoU.png        # Image showing IoU and Inference Speed on the KITTI (MOTS) dataset
├── Media                                 # Folder containing qualitative results from the paper
└── RGBF
│   └── correlation
│       └── correlatation.py              # Correlation file required to run LiteFlowNet
│   └── models/                           # Folder containing RGBF models from K=5 cross validation
│       └── CyberZoo/*                    # Folder containing RGBF models on the CyberZoo dataset
│       └── KITTI_Cars/*                  # Folder containing RGBF models on the KITTI (MOTS) dataset
│       └── KITTI_Pedestrians/*           # Folder containing RGBF models on the KITTI (MOTS Challenge) dataset
│   └── utils  
│       └── colorwheel.py                 # File to convert optical flow to color wheel representation
│       └── loss.py                       # File containing the loss functions used for training
└── RGBF_Fusion
│   └── correlation
│       └── correlatation.py                    
│   └── models/*
│   └── utils
│       └── colorwheel.py                      
│       └── loss.py  
└── U-Net
│   └── models/*
│   └── utils
│       └── loss.py  
└── U-Net_3D
│   └── models/*
│   └── utils
│       └── loss.py
└── U-Net_RNN
│   └── models/*
│   └── utils
│       └── convLSTM.py                   # File containing the structure of a convLSTM
│       └── convRNN.py                    # File containing the structure of a convRNN
│       └── loss.py
└── 
```

## Qualitative Results

![alt text](https://github.com/tommyvtran97/MAV-Segmentation/blob/master/Media/Cars_General_Marked_Red.png) 

&nbsp;
&nbsp;
&nbsp;

![alt text](https://github.com/tommyvtran97/MAV-Segmentation/blob/master/Media/CyberZoo_Normal_1_Marked_Red.png)


