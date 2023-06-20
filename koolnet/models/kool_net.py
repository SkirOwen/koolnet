import torch
import torch.nn.functional as F

from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
import lightning.pytorch as pl


class KoolNet(nn.Module):
	def __int__(self):
		super().__init__()
		self.c1 = nn.Sequential(
			nn.Conv2d(),
			nn.Tanh(),
			nn.MaxPool2d(),
		)

		self.c2 = nn.Sequential(
			nn.Conv2d(),
			nn.Tanh(),
			nn.MaxPool2d(),
		)
		self.c3 = nn.Sequential(
			nn.Conv2d(),
			nn.Tanh(),
		)
		self.c4 = nn.Sequential(
			nn.Conv2d(),
			nn.Tanh(),
		)
		self.c5 = nn.Sequential(
			nn.Conv2d(),
			nn.Tanh(),
			nn.MaxPool2d(),
		)

		self.fc1 = nn.Sequential(
			nn.Linear(),
			nn.Dropout(),
			nn.Tanh(),
		)

		self.fc2 = nn.Sequential(
			nn.Linear(),
			nn.Dropout(),
			nn.Tanh(),
		)

		self.fc3 = nn.Sequential(
			nn.Linear(),
			nn.Dropout(),
			nn.Tanh(),
		)

		self.out = nn.Sequential(
			nn.Linear(in_features=4000, out_features=2),
		)

		self.model = nn .Sequential(
			self.c1(),
			self.c2(),
			self.c3(),
			self.c4(),
			self.c5(),
			self.fc1(),
			self.fc2(),
			self.fc3(),
			self.fc4(),
		)

	def forward(self, x):
		return self.model(x)


def main():
	pass


if __name__ == "__main__":
	main()
