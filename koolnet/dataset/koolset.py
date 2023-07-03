import torch
from torch.utils.data import Dataset

from koolnet.data.preprocessing import data_window_mode


class Koolset(Dataset):
	def __init__(self, x, rel_pos, w):
		self.x = x
		self.rel_pos = rel_pos
		self.w = w

	def __len__(self):
		return len(self.x)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		image = torch.as_tensor(self.x[idx]).float()
		rel_pos_obst = torch.as_tensor(self.rel_pos[idx]).float()
		#

		return image, rel_pos_obst

	def get_window(self, idx):
		return torch.as_tensor(self.w[idx])
