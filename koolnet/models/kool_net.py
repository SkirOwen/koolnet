import torch
import torch.nn.functional as F

from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
import lightning.pytorch as pl


class KoolNet(nn.Module):
	def __init__(self, mode_nbr, dropout: float):
		super().__init__()

		self.mode_nbr = mode_nbr

		self.features = nn.Sequential(
			nn.Conv2d(self.mode_nbr, 64, kernel_size=11, stride=4, padding=2),
			nn.Tanh(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.Conv2d(64, 192, kernel_size=5, padding=2),
			nn.Tanh(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.Conv2d(192, 384, kernel_size=3, padding=1),
			nn.Tanh(inplace=True),
			nn.Conv2d(384, 256, kernel_size=3, padding=1),
			nn.Tanh(inplace=True),
			nn.Conv2d(256, 256, kernel_size=3, padding=1),
			nn.Tanh(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
		)

		self.conv = nn.Sequential(
			nn.Conv2d(self.mode_nbr, 96, kernel_size=(3, 3), stride=1),
			nn.Tanh(),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.Conv2d(96, 256, kernel_size=(5, 5), stride=2),
			nn.Tanh(),
			nn.MaxPool2d(kernel_size=5),
			nn.Conv2d(256, 384, kernel_size=(3, 3), stride=2),
			nn.Tanh(),
			nn.Conv2d(384, 384,	kernel_size=(3, 3), stride=4),
			nn.Tanh(),
			nn.Conv2d(384, 384, kernel_size=(3, 3), stride=4),
			nn.Tanh(),
			nn.MaxPool2d(kernel_size=3),
		)

		self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

		self.lin = nn.Sequential(
			nn.Dropout(p=dropout),
			nn.Linear(in_features=384, out_features=4096),
			nn.Tanh(),
			nn.Linear(in_features=4096, out_features=4096),
			nn.Tanh(),
			nn.Dropout(p=dropout),
			nn.Linear(in_features=4096, out_features=4096),
			nn.Tanh(),
			nn.Dropout(p=dropout),
			nn.Linear(in_features=4096, out_features=2),
		)

	def forward(self, x):
		x = self.conv(x)
		x = torch.flatten(x, 1)
		x = self.lin(x)
		return x


def main():
	pass


if __name__ == "__main__":
	main()
