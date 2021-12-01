""" 

Implementation of the U-Net encoder-decoder network architecture
wiht a recurrent module. The input of the network is a single RGB image as a 4D tensor
with the shape (N, C, H, W) --> (N, 3, H, W).

"""

import torch
import torch.nn as nn
from utils.convLSTM import *

class single_conv(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(single_conv, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		x = self.conv(x)

		return x

class down_mp_single(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(down_mp_single, self).__init__()
		self.mp_conv = nn.Sequential(
			nn.MaxPool2d(2),
			nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		x = self.mp_conv(x)

		return x


class up_single(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(up_single, self).__init__()
		self.upsample 	= nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
		self.conv 		= nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True)
		)

	def forward(self, x1, x2):
		x1 		= self.upsample(x1)
		diffY 	= x2.size()[2] - x1.size()[2]
		diffX	= x2.size()[3] - x1.size()[3]
		padding = nn.ConstantPad2d((diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2), 0.)
		x1 		= padding(x1)

		x 		= x2 + x1
		x 		= self.conv(x)

		return x


class out_single(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(out_single, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

	def forward(self, x):
		x = self.conv(x)

		return x


class Encoder1(nn.Module):
	def __init__(self, factor, sequence_size, in_channels=3):
		super(Encoder1, self).__init__()
		self.sequence_size = sequence_size

		self.input = single_conv(in_channels, int(64/factor))
		self.down1 = down_mp_single(int(64/factor), int(128/factor))
		self.down2 = down_mp_single(int(128/factor), int(256/factor))
		self.down3 = down_mp_single(int(256/factor), int(512/factor))
		self.down4 = down_mp_single(int(512/factor), int(512/factor))

	def forward(self, x):
		x1 = self.input(x)
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		x4 = self.down3(x3)
		x5 = self.down4(x4)
		
		return [x1, x2, x3, x4, x5]


class Decoder1(nn.Module):
	def __init__(self, factor, sequence_size=1):
		super(Decoder1, self).__init__()

		self.up1 = up_single(int(512/factor), int(256/factor))
		self.up2 = up_single(int(256/factor), int(128/factor))
		self.up3 = up_single(int(128/factor), int(64/factor))
		self.up4 = up_single(int(64/factor), int(64/factor))
		self.out = out_single(int(64/factor), 1)

	def forward(self, x1):
		x = self.up1(x1[4], x1[3])
		x = self.up2(x, x1[2])
		x = self.up3(x, x1[1])
		x = self.up4(x, x1[0])
		x = self.out(x)

		return x


class Recurrent(nn.Module):
	def __init__(self, factor, in_channels, sequence_size):
		super(Recurrent, self).__init__()
		self.sequence_size = sequence_size

		self.convlstm = ConvLSTM(input_dim=int(in_channels/factor), 
			hidden_dim=[int(in_channels/factor)], 
			kernel_size=(3, 3),
			num_layers=1, 
			batch_first=True, 
			bias=True, 
			return_all_layers=False
			)
		self.batch_relu = nn.Sequential(nn.BatchNorm2d(int(in_channels/factor)), nn.ReLU(inplace=True))
		self.states 	= None

	def forward(self, x, reset=False, reset_avg=False, detach=False):
	
		x = torch.reshape(x, (-1, self.sequence_size, x.shape[1], x.shape[2], x.shape[3]))
		
		if reset:
			if reset_avg and self.states is not None:
				o_state = self.states[0][0]
				h_state = self.states[0][1]
				if o_state.shape[0] == 1:
					self.states[0][0] = o_state.fill_(torch.mean(o_state))
					self.states[0][1] = h_state.fill_(torch.mean(h_state))
				else:
					self.states[0][0] = torch.cat(o_state.shape[0]*[torch.unsqueeze(torch.mean(o_state, 0), dim=0)])
					self.states[0][1] = torch.cat(h_state.shape[0]*[torch.unsqueeze(torch.mean(h_state, 0), dim=0)])
			else:
				self.states = None

		if self.states is not None and detach:
			self.states 	= [tuple(state.detach() for state in i) for i in self.states]

		out, self.states = self.convlstm(x, self.states)
		output 	= out[0].reshape(-1, x.shape[2], x.shape[3], x.shape[4])

		return self.batch_relu(output)


class UNet_LSTM(nn.Module):
	def __init__(self, factor, sequence_size=1):
		super(UNet_LSTM, self).__init__()
		self.sequence_size = sequence_size

		self.encoder1 	= Encoder1(factor, self.sequence_size)
		self.LSTM   	= Recurrent(factor, int(512), self.sequence_size)
		self.decoder1 	= Decoder1(factor)
		self.factor 	= factor

	def forward(self, x, sequence=None, reset=False, reset_avg=False, detach=False):
		x 		= self.encoder1(x)
		x[4] 	= self.LSTM(x[4], reset=reset, reset_avg=reset_avg, detach=detach)
		x 		= self.decoder1(x)
		x 		= torch.sigmoid(x)

		return x







