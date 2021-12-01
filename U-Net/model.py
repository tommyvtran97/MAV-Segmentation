""" 

Implementation of the U-Net encoder-decoder network architecture. 
The input of the network is a single RGB image as a 4D tensor
with the shape (N, C, H, W) --> (N, 3, H, W).

"""

import torch
import torch.nn as nn

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
	def __init__(self, factor, sequence_size=1, in_channels=3):
		super(Encoder1, self).__init__()

		self.input = single_conv(in_channels*sequence_size, int(64/factor))
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


class UNet(nn.Module):
	def __init__(self, factor, sequence_size):
		super(UNet, self).__init__()

		self.encoder1 	= Encoder1(factor, sequence_size)
		self.decoder1 	= Decoder1(factor)
		self.factor 	= factor

	def forward(self, x):
		x = self.encoder1(x)
		x = self.decoder1(x)
		x = torch.sigmoid(x)

		return x




