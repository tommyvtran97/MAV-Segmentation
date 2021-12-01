""" 

This script is used to train the U-Net encoder-decoder network architecture
with a recurrent module. The sequence length of a training sample can be 
adjusted. For example sequence_size = 20 means a sequence of 20 images. BP_window 
can be adjusted and determines the number of time steps to backpropagate
through time (BPTT).

"""

from modelRNN import UNet_RNN
from dataloader import *
from torch.utils.data import DataLoader
from utils.loss import Loss

import os
import sys
import csv
import datetime
import torch.optim as optim

for k in [1]:
	print("Network Training Cross Validation K=" + str(k) + " Initialized!")
	print("\n")

	# Initialize hyper parameters
	sequence_size   = 20
	batch_size      = 4
	epochs          = 200
	resolution      = (288, 224)
	k_fold          = k
	full_training	= False

	BP_window		= 5
	lr_rate         = 1e-4  

	path_to_data    = '../Dataset/CyberZoo/'

	# Initialize CUDA settings
	use_cuda        = True
	cuda            = torch.cuda.is_available() and use_cuda
	device          = torch.device('cuda' if cuda else 'cpu')
	torch.manual_seed(seed=0)

	# Initialize the model
	path_models = 'models/'
	now         = datetime.datetime.now()

	model       = UNet_RNN(2).to(device)

	path_models += model.__class__.__name__ + '_' + str(model.factor) + '_'
	path_models += '%02d%02d%04d' % (now.day, now.month, now.year)
	path_models += '_%02d%02d%02d' % (now.hour, now.minute, now.second)
	path_models += '/'

	if not os.path.exists(path_models): 
		os.makedirs(path_models)
		print('Weights stored at ' + path_models + '\n')

	# Preparing the dataset
	data = MAVDataset(path_to_data=path_to_data,
		batch_size=batch_size, 
		sequence_size=sequence_size, 
		resolution=resolution,
		k_fold=k_fold,  
		augmentation=True,
		augmentation_methods=[
			# 'Horizontal',
			# 'Rotation',
			# 'Shear',
			# 'Blur',
			# 'HSV'
			]
		)

	# Evaluate in real-time with batch = 1 and sequence_size = 1
	val_data = MAVDataset(path_to_data=path_to_data, 
		batch_size=1, 
		sequence_size=1, 
		resolution=resolution,
		k_fold=k_fold,
		validate=True,
		)
	dataloader = torch.utils.data.DataLoader(data, 
		batch_size=batch_size,
		collate_fn=data.custom_collate, 
		worker_init_fn=worker_init_fn,
		)
	# Evaluate in real-time with batch = 1
	val_dataloader = torch.utils.data.DataLoader(val_data,
		batch_size=1,
		collate_fn=data.custom_collate,
		worker_init_fn=worker_init_fn,
		)

	# Define loss function, optimizer and learning rate scheduler
	loss_criteria   = Loss()
	optimizer       = optim.Adam(model.parameters(), lr=lr_rate)
	lr_scheduler    = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[400], gamma=0.1)

	# Create CSV file for logging training data
	train_log       = open(path_models + "train_log.csv", 'w', newline='')
	train_writer    = csv.writer(train_log, delimiter=',')

	val_log         = open(path_models + "val_log.csv", 'w', newline='')
	val_writer      = csv.writer(val_log, delimiter=',')

	# Train the model
	best_loss       = 1.e6
	best_val_acc    = 0
	loss_seq		= 0
	loss_acc        = 0

	detach_states 	= False

	for epoch in range(epochs):
		train_loss  = 0
		IoU         = 0
		accuracy    = 0

		TP_ctr, TN_ctr, FP_ctr, FN_ctr  = 0, 0, 0, 0

		epoch_ctr = epoch

		# Shuffle data after every epoch
		data.shuffle()

		# Initialize training mode
		model.train()
		for batch_i, (imgs, targets, files) in enumerate(dataloader):
			
			if sequence_size > 1:
				folder_name = files[0][0][0].split("/")[5]
			else:
				folder_name = files[0][0].split("/")[5]

			imgs 	= imgs.reshape(batch_size, sequence_size, imgs.shape[1], imgs.shape[2], imgs.shape[3])
			targets = targets.reshape(batch_size, sequence_size, targets.shape[1], targets.shape[2], targets.shape[3])

			for T in range(sequence_size):
				img 		= imgs[:, T, :, :]
				target		= targets[:, T, :, :]

				if T == 0:
					x = model(img.to(device), reset=True)
					detach_states = False
				else:
					x = model(img.to(device), reset=False, detach=detach_states)
					detach_states = False

				loss = loss_criteria(x, target.to(device))
				loss_acc = loss_acc + loss

				loss_seq = loss_seq + loss.item()

				if (T+1) % BP_window == 0:
					optimizer.zero_grad()
					loss_acc.backward()
					optimizer.step()

					detach_states	= True
					loss_acc 		= 0

				# Calculate IoU and accuracy if segmentation of image is available
				for i in range(len(target)):
					if torch.sum(target[i]) != 0:
						TP = (np.logical_and(torch.round(x[i]).cpu() == 1, torch.round(target[i]).cpu() == 1)).sum().numpy()
						TN = (np.logical_and(torch.round(x[i]).cpu() == 0, torch.round(target[i]).cpu() == 0)).sum().numpy()
						FP = (np.logical_and(torch.round(x[i]).cpu() == 1, torch.round(target[i]).cpu() == 0)).sum().numpy()
						FN = (np.logical_and(torch.round(x[i]).cpu() == 0, torch.round(target[i]).cpu() == 1)).sum().numpy()

						TP_ctr      += TP
						TN_ctr      += TN
						FP_ctr      += FP
						FN_ctr      += FN

			train_loss += loss_seq / sequence_size
			loss_seq = 0

			print('Train Epoch {:04d} [{:03d}/{:03d} ({:03d}%)] Loss: {:.6f}'. format(epoch_ctr+1,
				batch_i,
				len(dataloader),
				int(100 * batch_i / len(dataloader)),
				train_loss / (batch_i + 1)),
				end='\r')

			folder_check = folder_name

		# Calculate IoU and accuracy at end of each epoch
		IoU         = round(TP_ctr / (TP_ctr + FP_ctr + FN_ctr), 4)
		accuracy    = round((TP_ctr + TN_ctr) / (TP_ctr + FP_ctr + FN_ctr + TN_ctr), 4)

		# Log the training 
		train_writer.writerow([epoch_ctr+1, train_loss / (batch_i + 1), IoU, accuracy])
		train_log.flush()

		if full_training:
			# Save best model on training dataset
			if train_loss / (batch_i + 1) < best_loss:
				for file in os.listdir(path_models):
					if file != "train_log.csv" and file != "val_log.csv":
						os.remove(path_models + file)
				model_name = path_models + "{:05d}_{:.4f}.pwf".format(epoch_ctr+1, train_loss / (batch_i+1))
				torch.save(model.state_dict(), model_name)
				best_loss = train_loss / (batch_i + 1)
		else:
			# Evaluate the performance on validation split
			val_train_loss  = 0
			val_IoU         = 0
			val_accuracy    = 0

			TP_val_ctr, TN_val_ctr, FP_val_ctr, FN_val_ctr  = 0, 0, 0, 0

			# Initialize evaluation mode
			model.eval()
			with torch.no_grad():
				for batchval_i, (imgs, targets, files) in enumerate(val_dataloader):

					if sequence_size > 1:
						folder_name = files[0][0].split("/")[5]     # changed this
					else:
						folder_name = files[0][0].split("/")[5]     

					# Reset the hidden state in LSTM after each sequence
					if batchval_i == 0 or folder_check != folder_name:
						x = model(imgs.to(device), reset=True)
					else:
						x = model(imgs.to(device), reset=False)

					loss = loss_criteria(x, targets.to(device))
					val_train_loss += loss.item()

					# Calculate IoU and accuracy if segmentation of image is available
					for i in range(len(targets)):
						if torch.sum(targets[i]) != 0:
							val_TP = (np.logical_and(torch.round(x[i]).cpu() == 1, torch.round(targets[i]).cpu() == 1)).sum().numpy()
							val_TN = (np.logical_and(torch.round(x[i]).cpu() == 0, torch.round(targets[i]).cpu() == 0)).sum().numpy()
							val_FP = (np.logical_and(torch.round(x[i]).cpu() == 1, torch.round(targets[i]).cpu() == 0)).sum().numpy()
							val_FN = (np.logical_and(torch.round(x[i]).cpu() == 0, torch.round(targets[i]).cpu() == 1)).sum().numpy()

							TP_val_ctr      += val_TP
							TN_val_ctr      += val_TN
							FP_val_ctr      += val_FP
							FN_val_ctr      += val_FN

					folder_check = folder_name

				# Calculate IoU and accuracy of the validation dataset
				val_IoU         = round(TP_val_ctr / (TP_val_ctr + FP_val_ctr + FN_val_ctr), 4)
				val_accuracy    = round((TP_val_ctr + TN_val_ctr) / (TP_val_ctr + FP_val_ctr + FN_val_ctr + TN_val_ctr), 4)

				# Log the training 
				val_writer.writerow([epoch_ctr+1, val_train_loss / (batchval_i + 1), val_IoU, val_accuracy])
				val_log.flush()

				# Save best model on validation dataset
				if val_IoU > best_val_acc:
					for file in os.listdir(path_models):
						if file != "train_log.csv" and file != "val_log.csv":
							os.remove(path_models + file)
					model_name = path_models + "{:05d}_{:.4f}.pwf".format(epoch_ctr+1, train_loss / (batch_i+1))
					torch.save(model.state_dict(), model_name)
					best_val_acc = val_IoU

		# Learning rate scheduler step
		lr_scheduler.step()

	print("\n")






