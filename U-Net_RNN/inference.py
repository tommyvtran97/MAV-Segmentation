""" 
	
This script is used to evaluate the runtime performance of
the U-Net encoder-decoder network architecture with a recurrent
module on a desktop GPU (NVIDIA GeForce RTX 2060).

"""

from modelRNN import UNet_RNN
from dataloader import *
from utils.loss import Loss

import numpy as np
import cv2 as cv
import os
import sys
import csv

# Initialize hyper parameters
sequence_size 	= 1
batch_size		= 1
resolution		= (288, 224)
k_fold			= 1

validate		= True

# Initialize CUDA settings
use_cuda 		= True
cuda 			= torch.cuda.is_available() and use_cuda
device 			= torch.device('cuda' if cuda else 'cpu')

# Load the model
gatenet 		= 'CyberZoo/CZ_RNN_S20_B5_K1'
load_model 		= 'models/' + gatenet + '/'
model 			= UNet_RNN(2, sequence_size).to(device)

if os.path.isdir(load_model):
	if len(os.listdir(load_model)) >= 1:
		for file in os.listdir(load_model):
			if file.split('.')[-1] == 'pwf':
				net = load_model + file
				model.load_state_dict(torch.load(net, map_location=device))
				print('Model restored from ' + net + '\n')

# Choose the dataset folder
dataset_folders = [
		# '../Dataset/KITTI_Cars/'
		# '../Dataset/KITTI_Pedestrians/'
		'../Dataset/CyberZoo/'
		]

# Evaluate the runtime performance
for ds in dataset_folders:
	print("Evaluating " + ds + " ...")

	path_to_data	= ds
	path_output		= ds + gatenet.split("/")[-1] + '/'

	data = MAVDataset(path_to_data=path_to_data,
		batch_size=batch_size, 
		sequence_size=sequence_size,
		resolution=resolution, 
		k_fold=k_fold,
		validate=validate,
		)
	dataloader = torch.utils.data.DataLoader(data,
		batch_size=batch_size,
		collate_fn=data.custom_collate,
		worker_init_fn=worker_init_fn,
		)

	# Initialize loggers
	starter, ender 	= torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
	repetitions 	= len(dataloader)
	timings			= np.zeros((repetitions,1))
	dummy_input		= torch.randn(sequence_size, 3, resolution[1], resolution[0], dtype=torch.float).to(device)

	# Warm Up GPU
	for _ in range(10):
		_ = model(dummy_input)

	# Measure runtime performance
	model.eval()
	with torch.no_grad():
		for k in range(1):
			for batch_i, (imgs, targets, files) in enumerate(dataloader):

				if sequence_size > 1:
					folder_name = files[0][0][0].split("/")[5]
				else:
					folder_name = files[0][0].split("/")[5]

				starter.record()

				x = model(imgs.to(device))

				ender.record()

				torch.cuda.synchronize()
				curr_time 			= starter.elapsed_time(ender)
				timings[batch_i] 	= curr_time

				folder_check = folder_name
			
mean_syn 	= np.sum(timings) / repetitions
std_syn 	= np.std(timings)

print("")
print("Time [ms]: \t", np.round(mean_syn, 4))
print("FPS: \t\t", np.round(1000/mean_syn, 4))
print("")


