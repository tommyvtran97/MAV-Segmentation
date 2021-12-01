""" 

This script is used to evaluate the performance of the 3D U-Net encoder-decoder network 
architecture. The performance is evaluated using the Intersection over Union (IoU)
metric. 

"""

from model import UNet_3D
from dataloader3D import *
from utils.loss import Loss

import numpy as np
import cv2 as cv
import os
import sys
import csv

# Initialize hyper parameters
sequence_size 	= 2
batch_size		= 1
resolution		= (288, 224)
k_fold			= 1

validate		= True
save_image		= True

# Initialize CUDA settings
use_cuda 		= True
cuda 			= torch.cuda.is_available() and use_cuda
device 			= torch.device('cuda' if cuda else 'cpu')

# Load the model
gatenet 		= 'CyberZoo/CZ_3D_2F_K1'
load_model 		= 'models/' + gatenet + '/'
model 			= UNet_3D(2, sequence_size).to(device)

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
		
# Evaluate the performance of the network on the dataset
for ds in dataset_folders:
	print("Evaluating " + ds + " ...")

	path_to_data	= ds
	path_output		= ds + gatenet.split("/")[-1] + '/'

	if not os.path.exists(path_output):
		os.makedirs(path_output)

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

# Create CSV file for logging the data
	data_log 	= open(path_output + "data_log.csv", 'w', newline='')
	data_writer = csv.writer(data_log, delimiter=',')

	TP_ctr, TN_ctr, FP_ctr, FN_ctr					= 0, 0, 0, 0
	TP_global, TN_global, FP_global, FN_global		= 0, 0, 0, 0

	batch_ctr, eval_loss, folder_check 	= 0, 0, ""
	loss_criteria 						= Loss()

	# Initialize evaluation mode
	model.eval()
	with torch.no_grad():
		for batch_i, (imgs, targets, files) in enumerate(dataloader):

			if sequence_size > 1:
				folder_name = files[0][0][0].split("/")[5]
			else:
				folder_name = files[0][0].split("/")[5]

			if (folder_check != folder_name and batch_i != 0):
				
				IoU 		= round(TP_ctr / (TP_ctr + FP_ctr + FN_ctr), 4)
				accuracy 	= round((TP_ctr + TN_ctr) / (TP_ctr + FP_ctr + FN_ctr + TN_ctr), 4)
				precision 	= round(TP_ctr / (TP_ctr + FP_ctr), 4)
				recall 		= round(TP_ctr / (TP_ctr + FN_ctr), 4)
				F1_score 	= round(TP_ctr / (TP_ctr + 0.5*(FP_ctr + FN_ctr)), 4)
				TPR 		= round(TP_ctr / (TP_ctr + FN_ctr), 4)
				TNR 		= round(TN_ctr / (TN_ctr + FP_ctr), 4)

				IoU_global 			= round(TP_global / (TP_global + FP_global + FN_global), 4)
				accuracy_global 	= round((TP_global + TN_global) / (TP_global + FP_global + FN_global + TN_global), 4)
				precision_global 	= round(TP_global / (TP_global + FP_global), 4)
				recall_global 		= round(TP_global / (TP_global + FN_global), 4)
				F1_score_global 	= round(TP_global / (TP_global + 0.5*(FP_global + FN_global)), 4)
				TPR_global 			= round(TP_global / (TP_global + FN_global), 4)
				TNR_global 			= round(TN_global / (TN_global + FP_global), 4)

				print("Folder: " + folder_check +  " - Evaluation Loss:", round(eval_loss / (batch_ctr), 4), "- IoU:", IoU)

				# Log the data
				data_writer.writerow([folder_check, eval_loss / (batch_ctr), IoU, precision, recall, F1_score, TPR, TNR, IoU_global, precision_global, recall_global, F1_score_global, TPR_global, TNR_global])
				data_log.flush()

				TP_ctr, TN_ctr, FP_ctr, FN_ctr 						= 0, 0, 0, 0
				batch_ctr, eval_loss 								= 0, 0

			x = model(imgs.to(device))

			folder_class = folder_name + '/'
			if not os.path.exists(path_output + folder_class):
				os.makedirs(path_output + folder_class)

			loss = loss_criteria(x, targets.to(device))
			eval_loss += loss.item()

			img = np.squeeze(x, axis=2)
			img = img.cpu().numpy().transpose((0, 2, 3, 1))

			image = cv.cvtColor(img[0]*255, cv.COLOR_GRAY2RGB)

			if save_image and sequence_size > 1:
				cv.imwrite(path_output + folder_class + files[0][-1][0].split('/')[-1], image)

			# Calculate IoU and accuracy if segmentation of image is available
			for i in range(len(targets)):
				if torch.sum(targets[i][0]) != 0:
					TP = (np.logical_and(torch.round(x[i][0]).cpu() == 1, torch.round(targets[i][0]).cpu() == 1)).sum().numpy()
					TN = (np.logical_and(torch.round(x[i][0]).cpu() == 0, torch.round(targets[i][0]).cpu() == 0)).sum().numpy()
					FP = (np.logical_and(torch.round(x[i][0]).cpu() == 1, torch.round(targets[i][0]).cpu() == 0)).sum().numpy()
					FN = (np.logical_and(torch.round(x[i][0]).cpu() == 0, torch.round(targets[i][0]).cpu() == 1)).sum().numpy()
				
					TP_ctr		+= TP
					TN_ctr		+= TN
					FP_ctr		+= FP
					FN_ctr		+= FN

					TP_global	+= TP
					TN_global	+= TN
					FP_global	+= FP
					FN_global	+= FN

			folder_check = folder_name
			
			batch_ctr 	+= 1

			if (batch_i == len(dataloader) - 1):
				if TP_ctr != 0:
					IoU 		= round(TP_ctr / (TP_ctr + FP_ctr + FN_ctr), 4)
					accuracy 	= round((TP_ctr + TN_ctr) / (TP_ctr + FP_ctr + FN_ctr + TN_ctr), 4)
					precision 	= round(TP_ctr / (TP_ctr + FP_ctr), 4)
					recall 		= round(TP_ctr / (TP_ctr + FN_ctr), 4)
					F1_score 	= round(TP_ctr / (TP_ctr + 0.5*(FP_ctr + FN_ctr)), 4)
					TPR 		= round(TP_ctr / (TP_ctr + FN_ctr), 4)
					TNR 		= round(TN_ctr / (TN_ctr + FP_ctr), 4)
				else:
					IoU, accuracy, precision, recall 	= 0, 0, 0, 0
					F1_score, TPR, TNR 					= 0, 0, 0
				IoU_global 			= round(TP_global / (TP_global + FP_global + FN_global), 4)
				accuracy_global 	= round((TP_global + TN_global) / (TP_global + FP_global + FN_global + TN_global), 4)
				precision_global 	= round(TP_global / (TP_global + FP_global), 4)
				recall_global 		= round(TP_global / (TP_global + FN_global), 4)
				F1_score_global 	= round(TP_global / (TP_global + 0.5*(FP_global + FN_global)), 4)
				TPR_global 			= round(TP_global / (TP_global + FN_global), 4)
				TNR_global 			= round(TN_global / (TN_global + FP_global), 4)

				print("Folder: " + folder_check +  " - Evaluation Loss:", round(eval_loss / (batch_ctr), 4), "- IoU:", IoU)

				# Log the data
				data_writer.writerow([folder_check, eval_loss / (batch_ctr), IoU, precision, recall, F1_score, TPR, TNR, IoU_global, precision_global, recall_global, F1_score_global, TPR_global, TNR_global])
				data_log.flush()

				TP_ctr, TN_ctr, FP_ctr, FN_ctr 						= 0, 0, 0, 0
				batch_ctr, eval_loss 								= 0, 0







