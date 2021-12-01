""" 

This script is used to evaluate the performance of the U-Net encoder-decoder network 
architecture with a recurrent module. The performance is evaluated using the 
Intersection over Union (IoU) metric. 

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
sequence_size 	= 20
batch_size		= 4
resolution		= (620, 186)
k_fold			= 1

validate		= True
save_image		= True		

# Initialize CUDA settings
use_cuda 		= True
cuda 			= torch.cuda.is_available() and use_cuda
device 			= torch.device('cuda' if cuda else 'cpu')

# Load the model
gatenet 		= 'KITTI_Cars/KC_RNN_S20_B5_K1'
load_model 		= 'models/' + gatenet + '/'
model 			= UNet_RNN(2).to(device)

if os.path.isdir(load_model):
	if len(os.listdir(load_model)) >= 1:
		for file in os.listdir(load_model):
			if file.split('.')[-1] == 'pwf':
				net = load_model + file
				model.load_state_dict(torch.load(net, map_location=device))
				print('Model restored from ' + net + '\n')

# Choose the dataset folder
dataset_folders = [
		'../Dataset/KITTI_Cars/'
		# '../Dataset/KITTI_Pedestrians/'
		# '../Dataset/CyberZoo/'
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

	loss_seq 							= 0

	# Initialize evaluation mode
	model.eval()
	with torch.no_grad():
		for batch_i, (imgs, targets, files) in enumerate(dataloader):

			if sequence_size > 1:
				folder_name = files[0][0][0].split("/")[5]
			else:
				folder_name = files[0][0].split("/")[5]

			imgs 	= imgs.reshape(batch_size, sequence_size, imgs.shape[1], imgs.shape[2], imgs.shape[3])
			targets = targets.reshape(batch_size, sequence_size, targets.shape[1], targets.shape[2], targets.shape[3])

			if (folder_check != folder_name and batch_i != 0):

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

				if TP_global != 0:
					IoU_global 			= round(TP_global / (TP_global + FP_global + FN_global), 4)
					accuracy_global 	= round((TP_global + TN_global) / (TP_global + FP_global + FN_global + TN_global), 4)
					precision_global 	= round(TP_global / (TP_global + FP_global), 4)
					recall_global 		= round(TP_global / (TP_global + FN_global), 4)
					F1_score_global 	= round(TP_global / (TP_global + 0.5*(FP_global + FN_global)), 4)
					TPR_global 			= round(TP_global / (TP_global + FN_global), 4)
					TNR_global 			= round(TN_global / (TN_global + FP_global), 4)
				else:
					IoU_global, accuracy_global,precision_global, recall_global = 0, 0, 0, 0
					F1_score_global, TPR_global, TNR_global 					= 0, 0, 0

				print("Folder: " + folder_check +  " - Evaluation Loss:", round(eval_loss / (batch_ctr), 4), "- IoU:", IoU)

				# Log the data
				data_writer.writerow([folder_check, eval_loss / (batch_ctr), IoU, precision, recall, F1_score, TPR, TNR, IoU_global, precision_global, recall_global, F1_score_global, TPR_global, TNR_global])
				data_log.flush()

				TP_ctr, TN_ctr, FP_ctr, FN_ctr 						= 0, 0, 0, 0
				batch_ctr, eval_loss 								= 0, 0

			folder_class = folder_name + '/'
			if not os.path.exists(path_output + folder_class):
				os.makedirs(path_output + folder_class)

			for T in range(sequence_size):
				img 		= imgs[:, T, :, :]
				target		= targets[:, T, :, :]

				if batch_i == 0 or folder_check != folder_name:
					x = model(img.to(device), reset=True)
				else:
					x = model(img.to(device), reset=True, reset_avg=False)

				loss = loss_criteria(x, target.to(device))
				loss_seq = loss_seq + loss.item()

				img = x.cpu().numpy().transpose((0, 2, 3, 1))

				image_1 = cv.cvtColor(img[0]*255, cv.COLOR_GRAY2RGB)
				image_2 = cv.cvtColor(img[1]*255, cv.COLOR_GRAY2RGB)
				image_3 = cv.cvtColor(img[2]*255, cv.COLOR_GRAY2RGB)
				image_4 = cv.cvtColor(img[3]*255, cv.COLOR_GRAY2RGB)

				if save_image and sequence_size == 1:
					cv.imwrite(path_output + folder_class + files[0][-1].split('/')[-1], image_1)
				else:
					cv.imwrite(path_output + folder_class + files[0][T][0].split('/')[-1], image_1)
					cv.imwrite(path_output + folder_class + files[1][T][0].split('/')[-1], image_2)
					cv.imwrite(path_output + folder_class + files[2][T][0].split('/')[-1], image_3)
					cv.imwrite(path_output + folder_class + files[3][T][0].split('/')[-1], image_4)

				# Calculate IoU and accuracy if segmentation of image is available
				for i in range(len(target)):
					if torch.sum(target[i]) != 0:
						TP = (np.logical_and(torch.round(x[i]).cpu() == 1, torch.round(target[i]).cpu() == 1)).sum().numpy()
						TN = (np.logical_and(torch.round(x[i]).cpu() == 0, torch.round(target[i]).cpu() == 0)).sum().numpy()
						FP = (np.logical_and(torch.round(x[i]).cpu() == 1, torch.round(target[i]).cpu() == 0)).sum().numpy()
						FN = (np.logical_and(torch.round(x[i]).cpu() == 0, torch.round(target[i]).cpu() == 1)).sum().numpy()

						TP_ctr		+= TP
						TN_ctr		+= TN
						FP_ctr		+= FP
						FN_ctr		+= FN

						TP_global	+= TP
						TN_global	+= TN
						FP_global	+= FP
						FN_global	+= FN

			eval_loss += loss_seq / sequence_size
			loss_seq = 0

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




