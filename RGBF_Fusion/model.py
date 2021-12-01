""" 

Implementation of the 2-stream U-Net encoder-decoder network architecture. 
The input of the network are 2 RGB images at [t-1, t]. The LiteFlowNet algorithm
is used to calculate the optical flow. One network is used to process the optical
flow maps and the other network to process the RGB images. Both modalities are fused
in the deepest layer of the encoder.

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


class conv_1D(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(conv_1D, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
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


class Encoder2(nn.Module):
	def __init__(self, factor, sequence_size=1, in_channels=3):
		super(Encoder2, self).__init__()

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

		self.conv0 = conv_1D(int(1024/factor), int(1024/factor))		# Added 1x1 Convolution after fusion
		self.conv1 = conv_1D(int(1024/factor), int(1024/factor))
		self.conv2 = conv_1D(int(512/factor), int(512/factor))
		self.conv3 = conv_1D(int(256/factor), int(256/factor))
		self.conv4 = conv_1D(int(128/factor), int(128/factor))
		
		self.up1 = up_single(int(1024/factor), int(512/factor))
		self.up2 = up_single(int(512/factor), int(256/factor))
		self.up3 = up_single(int(256/factor), int(128/factor))
		self.up4 = up_single(int(128/factor), int(64/factor))
		self.out = out_single(int(64/factor), 1)

	def forward(self, x1, y1):
		x = torch.cat((x1[4], y1[4]), dim=1)

		x = self.conv0(x)

		x = self.up1(x, self.conv1(torch.cat((x1[3], y1[3]), dim=1)))
		x = self.up2(x, self.conv2(torch.cat((x1[2], y1[2]), dim=1)))
		x = self.up3(x, self.conv3(torch.cat((x1[1], y1[1]), dim=1)))
		x = self.up4(x, self.conv4(torch.cat((x1[0], y1[0]), dim=1)))
		x = self.out(x)

		return x

class UNetFlow_2Stream(nn.Module):
	def __init__(self, factor, sequence_size, in_channels=3):
		super(UNetFlow_2Stream, self).__init__()

		self.encoder1 	= Encoder1(factor, in_channels=in_channels)
		self.encoder2	= Encoder2(factor)
		self.decoder1	= Decoder1(factor)
		self.factor 	= factor

	def forward(self, x, y):
		x1 = self.encoder1(x)
		y1 = self.encoder2(y)

		x = self.decoder1(x1, y1)
		x = torch.sigmoid(x)

		return x








