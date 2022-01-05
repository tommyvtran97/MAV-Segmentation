import matplotlib.pyplot as plt
import numpy as np 
import cv2 as cv
import glob as glob
import argparse

parser = argparse.ArgumentParser(description='Show Qualitative Results from The Paper')
parser.add_argument('--dataset', type=int,
                    help='A required integer positional, KITTI=1 and CyberZoo=2', default=1)

args = parser.parse_args()

if args.dataset == 1:
	resolution 	= (620, 186)
	dataset 	= 'KITTI_Cars'
	type_data 	= 'KC'
	k_fold 		= '2'

if args.dataset == 2:
	resolution 	= (288, 224)
	dataset 	= 'CyberZoo'
	type_data	= 'CZ'
	k_fold 		= '2'

threshold	= [150, 150, 150]
mask_color  = [0, 255, 0]

UNet_path		= sorted(glob.glob('Dataset/' + dataset + '/network/A - U-Net/' + type_data + '_UNet_' + '1F' + '_K' + str(k_fold) + "/*"))[:-1]
RNN_path		= sorted(glob.glob('Dataset/' + dataset + '/network/B - RNN/' + type_data + '_RNN_' + 'S20_B5' + '_K' + str(k_fold) + "/*"))[:-1]
LSTM_path		= sorted(glob.glob('Dataset/' + dataset + '/network/C - LSTM/' + type_data + '_LSTM_' + 'S20_B5' + '_K' + str(k_fold) + "/*"))[:-1]

folder_idx = 0

while True:

	if folder_idx == len(UNet_path):
		folder_idx = 0

	img_path_folder 	= sorted(glob.glob('Dataset/' + dataset + '/training/image/' + UNet_path[folder_idx].split('/')[-1] + "/*"))
	mask_path_folder 	= sorted(glob.glob('Dataset/' + dataset + '/training/mask/' + UNet_path[folder_idx].split('/')[-1] + "/*"))

	UNet_path_folder 	= sorted(glob.glob(UNet_path[folder_idx] + "/*"))
	RNN_path_folder 	= sorted(glob.glob(RNN_path[folder_idx] + "/*"))
	LSTM_path_folder 	= sorted(glob.glob(LSTM_path[folder_idx] + "/*"))

	length_list = [len(UNet_path_folder),len(RNN_path_folder), len(LSTM_path_folder)]

	for i in range(len(img_path_folder)):

		if i == min(length_list) - 1:
			break

		GT_background 		= cv.resize(cv.imread(img_path_folder[i+1]), resolution)
		GT_mask 			= cv.resize(cv.imread(mask_path_folder[i+1]), resolution)

		UNet_mask			= cv.imread(UNet_path_folder[i+1])
		RNN_mask			= cv.imread(RNN_path_folder[i+1])
		LSTM_mask			= cv.imread(LSTM_path_folder[i+1])

		GT_mask[np.all(GT_mask > threshold, axis=-1)] = mask_color
		UNet_mask[np.all(UNet_mask > threshold, axis=-1)] = mask_color
		RNN_mask[np.all(RNN_mask > threshold , axis=-1)] = mask_color
		LSTM_mask[np.all(LSTM_mask > threshold, axis=-1)] = mask_color

		GT_added_mask 		= cv.addWeighted(GT_background, 1.0, GT_mask, 0.6, 0)	
		UNet_added_mask 	= cv.addWeighted(GT_background, 1.0, UNet_mask, 0.6, 0)	
		RNN_added_mask 		= cv.addWeighted(GT_background, 1.0, RNN_mask, 0.6, 0)	
		LSTM_added_mask 	= cv.addWeighted(GT_background, 1.0, LSTM_mask, 0.6, 0)	

		if args.dataset == 1:
			concat_output = np.concatenate((GT_background, GT_added_mask, UNet_added_mask, RNN_added_mask, LSTM_added_mask), axis=0)
		if args.dataset == 2:
			concat_output = np.concatenate((GT_background, GT_added_mask, UNet_added_mask, RNN_added_mask, LSTM_added_mask), axis=1)

		cv.imshow(dataset + ': ' + UNet_path[folder_idx].split('/')[-1], concat_output)

		k = cv.waitKey(10) & 0xFF

		if k == 27:
			cv.destroyAllWindows()
			break
		if k == 61:
			cv.destroyAllWindows()
			folder_idx = folder_idx + 1
			break
		if k == 45:
			cv.destroyAllWindows()
			folder_idx = folder_idx - 1
			break

	if k == 27:
		cv.destroyAllWindows()
		break

