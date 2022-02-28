# Semantic Segmentation over Time using Deep Neural Networks

Semantic segmentation methods have been developed and applied to single images for object segmentation. However, for robotic applications such as high-speed agile Micro Air Vehicles (MAVs) in Autonomous Drone Racing (ADR), it is more interesting to consider temporal information as video images are correlated over time. In this paper, we evaluate the performance of state-of-the-art methods such as Recurrent Neural Networks (RNNs), 3D Convolutional Neural Networks (CNNs), and optical flow for video semantic segmentation in terms of accuracy and inference speed on three datasets with different camera motion configurations. The results show that using an RNN with convolutional operators outperforms all methods on the KITTI (MOTS) and CyberZoo dataset with 3 degrees of freedom (DoF) and 6 DoF, respectively. The inference speed was measured on the CyberZoo dataset achieving 321 Hz on an NVIDIA Geforce RTX 2060 GPU and 30 Hz on an NVIDIA Jetson TX2 mobile computer.

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

Download the datasets from the provided links. 

|#|Datasets|Download|
|---|----|-----|
|1|CyberZoo|[Link](https://drive.google.com/file/d/1fSv9Jqwge47XaM-f6HYxepWwz12mQ78-/view?usp=sharing)|
|2|KITTI MOTS|[Link](https://drive.google.com/file/d/1PTn7-sze5NqKp9KPQy5uaVQI6kWsEGYh/view?usp=sharing)|
|3|KITTI MOTS Challenge|[Link](https://drive.google.com/file/d/1Q1ispTWUObIiN_NQyAVEXi_IdQC6cNrV/view?usp=sharing)|

Additionally:
* Download the images for the KITTI MOTS dataset from [Link](http://www.cvlibs.net/download.php?file=data_tracking_image_2.zip) and add the training images to the directory `KITTI_Cars/training/image/`.
* Download the training dataset for the CyberZoo dataset from [Link](https://doi.org/10.4121/19042235.v1) and add them to the directory `CyberZoo/training/`.

If done correctly, the folder structure should be similar as shown below.

```
├── Dataset
│   └── CyberZoo
│       └── network        
│       └── training
│           └── horizontal
│           └── image
│           └── magnitude
│           └── mask
│           └── vertical
│   └── KITTI_Cars/*
│   └── KITTI_Pedestrians/*  
├── Figures
├── Media                                
└── RGBF
└── RGBF_Fusion
└── U-Net
└── U-Net_3D
└── U-Net_RNN
```

## Usage 
The content consists of 5 networks; RGBF, RGBF_Fusion, U-Net, U-Net_3D, and U-Net_RNN. We show an example for the U-Net_RNN folder, the same instructions hold for the other folders. The training settings can be changed in `RNN_train.py`. To train the network run:

```
cd U-Net_RNN
python RNN_train.py
```

To show the results of U-Net, RNN and LSTM. Use '1' for the KITTI dataset and '2' for the CyberZoo dataset:

```
python results.py --dataset 1
```

The results below show from top to bottom: RGB - GT - U-Net - RNN - LSTM

![alt text](https://github.com/tommyvtran97/MAV-Segmentation/blob/master/Media/Results.png)

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


