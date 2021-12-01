""" 

This script is used to properly load the data from the KITTI MOTS
dataset for the two-stream U-Net encoder-decoder network architecture. A
K-Fold cross validation is applied with K=5 with a 80/20 split. 

"""

import torch.utils.data as data
import matplotlib.pyplot as plt
import torch 
import math
import numpy as np 
import glob as glob
import cv2 as cv
import random
import os

# Define custom function to prepare the dataset
def resize_image(img, dim, inter=cv.INTER_AREA):
	resized_image = cv.resize(img, dim, interpolation=inter)

	return resized_image

def random_affine(img, mask, mag, hor, ver, random_number, degrees=(0, 0), translate=(0, 0), scale=(1, 1), shear=(0, 0)):
	height 	= img.shape[1] 
	width 	= img.shape[0] 

	# Rotation and Scale
	R = np.eye(3)
	a = random_number * (degrees[1] - degrees[0]) + degrees[0]
	s = random_number * (scale[1] - scale[0]) + scale[0]
	R[:2] = cv.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

	# Translation
	T = np.eye(3)
	T[0, 2] = (random_number * 2 - 1) * translate[0] * img.shape[0] 
	T[1, 2] = (random_number * 2 - 1) * translate[1] * img.shape[1] 

	# Shear
	S = np.eye(3)
	S[0, 1] = math.tan((random_number * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)
	S[1, 0] = math.tan((random_number * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)

	# Rotation matrix
	M = S @ T @ R  

	img 	= cv.warpPerspective(img, M, dsize=(height, width), flags=cv.INTER_LINEAR, borderValue=(127.5, 127.5, 127.5))
	mask 	= cv.warpPerspective(mask, M, dsize=(height, width), flags=cv.INTER_LINEAR, borderValue=(0., 0.))
	mask 	= np.expand_dims(mask, axis=2)

	if mag is not None: 
		mag 	= cv.warpPerspective(mag, M, dsize=(height, width), flags=cv.INTER_LINEAR, borderValue=(127.5, 127.5, 127.5))
		mag 	= np.expand_dims(mag, axis=2)
	if hor is not None:
		hor 	= cv.warpPerspective(hor, M, dsize=(height, width), flags=cv.INTER_LINEAR, borderValue=(127.5, 127.5, 127.5))
		hor 	= np.expand_dims(hor, axis=2)
	if ver is not None:
		ver 	= cv.warpPerspective(ver, M, dsize=(height, width), flags=cv.INTER_LINEAR, borderValue=(127.5, 127.5, 127.5))
		ver 	= np.expand_dims(ver, axis=2)

	return img, mask, mag, hor, ver

def motion_blur(img, mask, max_size):
	angle 	= np.random.random_sample() * 360.0
	size 	= np.random.randint(low=5, high=max_size)

	# Create, rotate, and normalize horizontal kernel
	k 					= np.zeros((size, size), dtype=np.float32)
	k[(size-1)// 2, :] 	= np.ones(size, dtype=np.float32)
	k 					= cv.warpAffine(k, cv.getRotationMatrix2D((size / 2 -0.5 , size / 2 -0.5 ), angle, 1.0), (size, size))  
	k 					= k * (1.0 / np.sum(k))        

	# Apply the kernel
	img = cv.filter2D(img, -1, k) 

	return img, mask

def prepare_batch_images(imgs, mag, hor, ver, cat_type):

	imgs = np.stack(imgs)
	imgs = imgs.reshape(-1, imgs.shape[2], imgs.shape[3], imgs.shape[4]) 					# Reorder shape (B,D,H,W,C) --> (B*D,H,W,C)
	imgs = imgs.transpose(0, 3, 1, 2)														# Reorder shape (B*D,H,W,C) --> (B*D,C,H,W)
	imgs = np.ascontiguousarray(imgs, dtype=np.float32)
	imgs /= 255.																			# Normalize values between [0,1]																		

	if cat_type == "M" or cat_type == "UVM":
		mag = np.stack(mag)
		mag = mag.reshape(-1, mag.shape[2], mag.shape[3], mag.shape[4]) 						
		mag = mag.transpose(0, 3, 1, 2)															
		mag = np.ascontiguousarray(mag, dtype=np.float32)										
		mag /= 255.																				

	if cat_type == "UV" or cat_type == "UVM":
		hor = np.stack(hor)
		hor = hor.reshape(-1, hor.shape[2], hor.shape[3], hor.shape[4]) 						
		hor = hor.transpose(0, 3, 1, 2)															
		hor = np.ascontiguousarray(hor, dtype=np.float32)										
		hor /= 255.																				

		ver = np.stack(ver)
		ver = ver.reshape(-1, ver.shape[2], ver.shape[3], ver.shape[4]) 						
		ver = ver.transpose(0, 3, 1, 2)															
		ver = np.ascontiguousarray(ver, dtype=np.float32)										
		ver /= 255.																				
																				
	reshape_imgs = np.reshape(imgs, (-1, 2, imgs.shape[1], imgs.shape[2], imgs.shape[3]))

	concat_input = []
	for i in range(len(reshape_imgs)):
		if cat_type == 'M':
			rgb_cat = np.expand_dims(np.concatenate((mag[i], reshape_imgs[i][-1]), axis=0), axis=0)
		elif cat_type == 'UV':
			rgb_cat = np.expand_dims(np.concatenate((hor[i], ver[i], reshape_imgs[i][-1]), axis=0), axis=0)
		elif cat_type == 'UVM':
			rgb_cat = np.expand_dims(np.concatenate((hor[i], ver[i], mag[i], reshape_imgs[i][-1]), axis=0), axis=0)
		elif cat_type == 'RGB':
			rgb_cat = np.expand_dims(np.concatenate((reshape_imgs[i][-2], reshape_imgs[i][-1]), axis=0), axis=0)

		if i == 0:
			concat_input = rgb_cat
		else:
			concat_input = np.concatenate((concat_input, rgb_cat), axis=0)

	return(torch.from_numpy(concat_input))

def prepare_batch_targets(imgs, cat_type):
	imgs = np.stack(imgs)
	imgs = imgs.reshape(-1, imgs.shape[2], imgs.shape[3], imgs.shape[4])					# Reorder shape (B,D,H,W,C) --> (B*D,H,W,C)
	imgs = imgs.transpose(0, 3, 1, 2)														# Reorder shape (B*D,H,W,C) --> (B*D,C,H,W)
	imgs = np.ascontiguousarray(imgs, dtype=np.float32)
	imgs /= 255.																			# Normalize values between [0,1]

	target_imgs 	= np.reshape(imgs, (-1, 2, imgs.shape[1], imgs.shape[2], imgs.shape[3]))

	reshape_imgs 	= []
	for i in range(len(target_imgs)):
		if i == 0:
			reshape_imgs = np.expand_dims(target_imgs[i][-1], axis=0)
		else:
			reshape_imgs = np.concatenate((reshape_imgs, np.expand_dims(target_imgs[i][-1], axis=0)), axis=0)

	return(torch.from_numpy(reshape_imgs))

def worker_init_fn(worker_id):
	np.random.seed(np.random.get_state()[1][0] + worker_id)

def prepare_dataset(path_to_data, batch_size, sequence_size, validate, k_fold):
	img_list	= np.array([])
	tar_list 	= np.array([])
	mag_list	= np.array([])
	hor_list	= np.array([])
	ver_list	= np.array([])

	image_data		= np.array([])
	target_data 	= np.array([])
	magnitude_data 	= np.array([])
	horizontal_data	= np.array([])
	vertical_data	= np.array([])

	# Switch between training and validation
	if not validate:
		train = True
	else:
		train = False

	# Create subset of training data for K-Fold cross validation with K=5
	if path_to_data.split('/')[-2] == 'KITTI_Cars':
		if k_fold == 0:
			k_subset	= ['0002', '0006', '0007', '0008', '0010', '0013', '0014', '0016', '0018']
		if k_fold == 1:
			k_subset	= ['0000', '0001', '0002', '0003', '0004', '0005']
		if k_fold == 2:
			k_subset	= ['0007', '0008', '0011', '0012']
		if k_fold == 3:
			k_subset	= ['0009', '0010', '0013', '0017']
		if k_fold == 4:
			k_subset	= ['0006', '0014', '0015', '0020']
		if k_fold == 5:
			k_subset	= ['0016', '0018', '0019']

	if path_to_data.split('/')[-2] == 'KITTI_Pedestrians':
		if k_fold == 1:
			k_subset	= ['0001','0006', '0007']
		if k_fold == 2:
			k_subset	= ['0002','0008']
		if k_fold == 3:
			k_subset	= ['0003','0009']
		if k_fold == 4:
			k_subset	= ['0004','0010']
		if k_fold == 5:
			k_subset	= ['0005','0011']

	if path_to_data.split('/')[-2] == 'CyberZoo':
		if k_fold == 0:
			k_subset 	= []
		if k_fold == 1:
			k_subset	= ['0001', '0002', '0003', '0005']
		if k_fold == 2:
			k_subset	= ['0006', '0007', '0008', '0009', '0012', '0014', '0015']
		if k_fold == 3:
			k_subset	= ['0010','0016', '0017', '0022']
		if k_fold == 4:
			k_subset	= ['0004','0011', '0013', '0018', '0021', '0024']
		if k_fold == 5:
			k_subset	= ['0019', '0020', '0023', '0025', '0026']

	if path_to_data.split('/')[-2] == 'AlphaPilot':
		if k_fold == 0:
			k_subset 	= []
		if k_fold == 1:
			k_subset	= ['0001','0002', '0009', '0010']
		if k_fold == 2:
			k_subset	= ['0003', '0008']
		if k_fold == 3:
			k_subset	= ['0004','0005', '0015']
		if k_fold == 4:
			k_subset	= ['0006','0012', '0013']
		if k_fold == 5:
			k_subset	= ['0007','0011', '0014']
			
	# Prepare the dataset
	data_sequence	= sorted(glob.glob(path_to_data + 'training/' + "image/" + "*"))
	counter 		= 0

	for i in range(len(data_sequence)):
		if data_sequence[i].split('/')[-1] in k_subset:
			training 	= False
			validation 	= True

		if data_sequence[i].split('/')[-1] not in k_subset:
			training	= True
			validation 	= False

		if (validate and validation) or  (train and training):
			images 		= sorted(glob.glob(data_sequence[i] + "/" + "*"))
			targets 	= sorted(glob.glob(data_sequence[i].replace("image", "mask") + "/" + "*"))
			magnitude 	= sorted(glob.glob(data_sequence[i].replace("image", "magnitude") + "/" + "*"))
			horizontal 	= sorted(glob.glob(data_sequence[i].replace("image", "horizontal") + "/" + "*"))
			vertical 	= sorted(glob.glob(data_sequence[i].replace("image", "vertical") + "/" + "*"))
			
			# Temporal overlap of the images
			for j in range(0, len(images), 1):

				img_list 	= []
				tar_list	= []
				mag_list 	= []
				hor_list	= []
				ver_list 	= []

				for k in range(sequence_size):
					if counter == 0 and j == 0:
						image_data 	= np.append(image_data, images[j+k])
						target_data = np.append(target_data, targets[j+k])

						if k == 0:
							magnitude_data  = np.append(magnitude_data, magnitude[j])
							horizontal_data = np.append(horizontal_data, horizontal[j])
							vertical_data  	= np.append(vertical_data, vertical[j])
							
					else:
						img_list 	= np.append(img_list, images[j+k])
						tar_list	= np.append(tar_list, targets[j+k])

						if k == 0:
							mag_list = np.append(mag_list, magnitude[j])
							hor_list = np.append(hor_list, horizontal[j])
							ver_list = np.append(ver_list, vertical[j])

					if k == sequence_size - 1 and (counter != 0 or j != 0):
						image_data 		= np.vstack((image_data, img_list))
						target_data		= np.vstack((target_data, tar_list))
						magnitude_data  = np.vstack((magnitude_data, mag_list))
						horizontal_data = np.vstack((horizontal_data, hor_list))
						vertical_data  	= np.vstack((vertical_data, ver_list))

				# Check that images are not out of bounds and ensure equal batch size = 4
				if j == ((len(images) - sequence_size) - ((len(images) - (sequence_size - 1)) % 4)):
					break

			counter += 1

	data = list(zip(image_data, target_data, magnitude_data, horizontal_data, vertical_data))

	return data


class MAVDataset(data.Dataset):
	def __init__(self, path_to_data, batch_size, sequence_size, cat_type, resolution=None, k_fold=1, validate=False, augmentation=False, augmentation_methods=[]):
		self.path_to_data 			= path_to_data
		self.batch_size				= batch_size
		self.sequence_size 			= sequence_size
		self.cat_type				= cat_type
		self.resolution 			= resolution
		self.k_fold					= k_fold
		self.validate				= validate
		self.augmentation 			= augmentation
		self.augmentation_methods 	= augmentation_methods

		self.data = prepare_dataset(self.path_to_data, self.batch_size, self.sequence_size, self.validate, self.k_fold)

	def __getitem__(self, index):
		while True:
			img_stack 			= np.array([])
			mask_stack			= np.array([])
			magnitude_stack		= np.array([])
			horizontal_stack	= np.array([])
			vertical_stack		= np.array([])
			name_stack			= np.array([])

			stop = False

			aug_horizontal 	= False
			aug_rotation 	= False
			aug_shear 		= False
			aug_hsv			= False

			if 'Horizontal' in self.augmentation_methods and random.random() > 0.5:
				aug_horizontal 		= True

			if 'Rotation' in self.augmentation_methods and random.random() > 0.5:
				aug_rotation 		= True
				random_number	 	= random.random()

			if 'Shear' in self.augmentation_methods and random.random() > 0.5:
				aug_horizontal 		= True
				random_number	 	= random.random()

			if 'HSV' in self.augmentation_methods and random.random() > 0.5:
				aug_hsv 			= True
				random_number		= random.random()

			for i in range(self.sequence_size):
				mask, mag, hor, ver = None, None, None, None

				img 	= cv.imread(self.data[index][0][i])

				if os.path.isfile(self.data[index][1][i]):
					mask = cv.imread(self.data[index][1][i], cv.IMREAD_GRAYSCALE)

				# Reading optical flow images
				if self.cat_type == "M" or self.cat_type == "UVM":
					if os.path.isfile(self.data[index][2][0]):
						mag = cv.imread(self.data[index][2][0], cv.IMREAD_GRAYSCALE)

				if self.cat_type == "UV" or self.cat_type == "UVM":
					if os.path.isfile(self.data[index][3][0]):
						hor = cv.imread(self.data[index][3][0], cv.IMREAD_GRAYSCALE)

					if os.path.isfile(self.data[index][4][0]):
						ver = cv.imread(self.data[index][4][0], cv.IMREAD_GRAYSCALE)

				# Resize image to specified resolution W x H
				img = resize_image(img, self.resolution)
				if mask is not None:
					mask = resize_image(mask, self.resolution)
				if mag is not None:
					mag = resize_image(mag, self.resolution)
				if hor is not None:
					hor = resize_image(hor, self.resolution)
				if ver is not None:
					ver = resize_image(ver, self.resolution)

				# Reshape mask from H x W --> H x W x C
				if mask is not None:
					mask = mask.reshape((mask.shape[0], mask.shape[1], 1))
				if mag is not None:
					mag = mag.reshape((mag.shape[0], mag.shape[1], 1))
				if hor is not None:
					hor = hor.reshape((hor.shape[0], hor.shape[1], 1))
				if ver is not None:
					ver = ver.reshape((ver.shape[0], ver.shape[1], 1))

				# DATA AUGMENTATION: Horizontal flips
				if self.augmentation and aug_horizontal:
					if 'Horizontal' in self.augmentation_methods:
						img = np.fliplr(img)
						mask = np.fliplr(mask)

						if mag is not None:
							mag = np.fliplr(mag)
						if hor is not None:
							hor = np.fliplr(hor)
						if ver is not None:
							ver = np.fliplr(ver)

				# DATA AUGMENTATION: Affine transformation
				if self.augmentation and (aug_rotation or aug_shear):

					rotation 	= (0, 0)
					translation = (0., 0.)
					scale 		= (1., 1.)
					shear 		= (0, 0)

					if 'Rotation' in self.augmentation_methods:
						rotation = (-15, 15)
					if 'Translation' in self.augmentation_methods:
						translation = (0.1, 0.1)
					if 'Scale' in self.augmentation_methods:
						scale = (0.25, 1.75)
					if 'Shear' in self.augmentation_methods:
						shear = (-10, 10)

					img, mask, mag, hor, ver = random_affine(img, mask, mag, hor, ver, random_number, degrees=rotation, translate=translation, scale=scale, shear=shear)

				# DATA AUGMENTATION: Artificial motion blur
				if self.augmentation:
					if 'Blur' in self.augmentation_methods and random.random() > 0.5:
						img, mask = motion_blur(img, mask, 15)

				# DATA AUGMENTATION: HSV hue, intensity, and saturation (+-50%)
				if self.augmentation and aug_hsv:
					img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
					H = img_hsv[:, :, 0].astype(np.float32)
					S = img_hsv[:, :, 1].astype(np.float32)
					V = img_hsv[:, :, 2].astype(np.float32)

					a = 1
					fraction = 0.15
					if random_number < 1.: a = (random_number * 2 - 1) * fraction + 1 
					H = np.clip(H * a, a_min=0, a_max=179)
					a = 1
					fraction = 0.5
					if random_number < 0.5: a = (random_number * 2 - 1) * fraction + 1 
					S = np.clip(S * a, a_min=0, a_max=255)
					a = 1
					fraction = 0.5
					if random_number < 0.5: a = (random_number * 2 - 1) * fraction + 1 
					V = np.clip(V * a, a_min=0, a_max=255)

					img_hsv[:, :, 0] = H.astype(np.uint8)
					img_hsv[:, :, 1] = S.astype(np.uint8)
					img_hsv[:, :, 2] = V.astype(np.uint8)
					img = cv.cvtColor(img_hsv, cv.COLOR_HSV2BGR)

				# Initialize first image
				if i == 0:
					img_temp 	= np.array([img])
					mask_temp 	= np.array([mask])

					if mag is not None:
						mag_temp 	= np.array([mag])
					if hor is not None:
						hor_temp 	= np.array([hor])
					if ver is not None:
						ver_temp 	= np.array([ver])

					name_temp	= np.array([self.data[index][0][i]])
				else:
					img_stack 	= np.vstack((img_temp, [img]))
					mask_stack 	= np.vstack((mask_temp, [mask])) 
					name_stack 	= np.vstack((name_temp, [self.data[index][0][i]]))

					img_temp 	= img_stack
					mask_temp	= mask_stack
					name_temp 	= name_stack

				if self.sequence_size == 1:
					img_stack 	= img_temp
					mask_stack 	= mask_temp
					name_stack 	= name_temp

				if self.sequence_size == 2:
					if mag is not None:
						mag_stack 	= mag_temp
					if hor is not None:
						hor_stack 	= hor_temp
					if ver is not None:
						ver_stack 	= ver_temp

				if i == self.sequence_size - 1:
					stop = True
					break
			if stop:
				break

		# Convert image from BGR to RGB 
		img_stack = np.array([cv.cvtColor(item, cv.COLOR_BGR2RGB) for item in img_stack])

		if mag is None:
			mag_stack = None
		if hor is None:
			hor_stack = None
		if ver is None:
			ver_stack = None

		return img_stack, mask_stack, mag_stack, hor_stack, ver_stack, name_stack

	def __len__(self):
		return(len(self.data))

	def custom_collate(self, batch):
		imgs 	= [item[0] for item in batch]
		mask 	= [item[1] for item in batch]
		mag 	= [item[2] for item in batch]
		hor 	= [item[3] for item in batch]
		ver 	= [item[4] for item in batch]
		names 	= [item[5] for item in batch]

		return prepare_batch_images(imgs, mag, hor, ver, self.cat_type), prepare_batch_targets(mask, self.cat_type), names

	def shuffle(self, flag=True):
		np.random.shuffle(self.data if flag else self.data)