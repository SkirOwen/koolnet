import lightning.pytorch as pl
from torch.utils.data import DataLoader

from torchvision import transforms

from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS


class KoolDataModule(pl.LightningDataModule):
	def __init__(self, batch_size, koolset_train, koolset_val, koolset_test, koolset_predict):
		super().__init__()
		self.save_hyperparameters()

		self.batch_size = batch_size

		self.koolset_train = koolset_train
		self.koolset_val = koolset_val
		self.koolset_test = koolset_test
		self.koolset_predict = koolset_predict

	def train_dataloader(self) -> TRAIN_DATALOADERS:
		return DataLoader(
			self.koolset_train,
			batch_size=self.batch_size,
			pin_memory=True,
			shuffle=True,
		)

	def val_dataloader(self) -> EVAL_DATALOADERS:
		return DataLoader(
			self.koolset_val,
			batch_size=self.batch_size,
			pin_memory=True,
		)

	def test_dataloader(self) -> EVAL_DATALOADERS:
		return DataLoader(
			self.koolset_test,
			batch_size=self.batch_size,
			pin_memory=True,
		)

	def predict_dataloader(self) -> EVAL_DATALOADERS:
		return DataLoader(
			self.koolset_predict,
			batch_size=self.batch_size,
			pin_memory=True,
		)
