import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class DiceLoss(nn.Module):
	def forward(self, x, y):
		smooth = 1.
		x_flat = x.reshape(-1)
		y_flat = y.reshape(-1)
		return 1 - ((2. * (x_flat * y_flat).sum() + smooth) /
					(x_flat.sum() + y_flat.sum() + smooth))

class FocalLoss(nn.Module):
	def __init__(self, alpha=1, gamma=2, size_average=True):
		super(FocalLoss, self).__init__()
		self.alpha = alpha
		self.gamma = gamma
		self.size_average = size_average
		self.BCE = nn.BCELoss()

	def forward(self, x, y):
		BCE_loss = self.BCE(x, y)
		pt = torch.exp(-BCE_loss)
		loss = self.alpha * torch.pow((1 - pt), self.gamma) * BCE_loss
		return  loss.mean() if self.size_average else loss.sum()

class Loss(nn.Module):
	def __init__(self):
		super(Loss, self).__init__()
		self.BCE = nn.BCELoss()
		self.Dice = DiceLoss()
		self.Focal = FocalLoss()

	def forward(self, x, y):
		return 2. * self.Dice(x, y) + self.Focal(x, y)

