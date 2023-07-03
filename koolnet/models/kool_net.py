from __future__ import annotations

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
import seaborn as sns

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import ModelSummary, RichModelSummary
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import RichProgressBar, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger

from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.functional import r2_score

from sklearn.model_selection import train_test_split

from tqdm import tqdm

from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT

from koolnet import logger
from koolnet import RANDOM_SEED
from koolnet.data.preprocessing import data_window_mode
from koolnet.dataloading.koolload import KoolDataModule
from koolnet.dataset.koolset import Koolset
from koolnet.utils import load_h5
from koolnet.utils.metrics import avg_rel_iou
from koolnet.utils.plotting import plot_multiple, plot_pred_obs_dist
from koolnet.models.predict import chain_multiple


class KoolNet(pl.LightningModule):
	def __init__(self, mode_nbr: int, dropout: float = 0.17, lr: float = 1e-5):
		super().__init__()
		self.save_hyperparameters()
		self.mode_nbr = mode_nbr
		self.lr = lr

		self.conv = nn.Sequential(
			nn.Conv2d(self.mode_nbr, 96, kernel_size=(3, 3), stride=1, padding=2),
			nn.Tanh(),
			nn.MaxPool2d(kernel_size=3, stride=2),

			nn.Conv2d(96, 256, kernel_size=(5, 5), padding=2),
			nn.Tanh(),
			nn.MaxPool2d(kernel_size=3, stride=1),

			nn.Conv2d(256, 384, kernel_size=(3, 3), padding=1),
			nn.Tanh(),

			nn.Conv2d(384, 384,	kernel_size=(3, 3), padding=1),
			nn.Tanh(),

			nn.Conv2d(384, 384, kernel_size=(3, 3), padding=1),
			nn.Tanh(),
			nn.MaxPool2d(kernel_size=3),
		)

		self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

		self.regressor = nn.Sequential(
			nn.Dropout(p=dropout),
			# mult the input by the size of the AdaptiveAvgPool
			nn.Linear(in_features=384 * 6 * 6, out_features=4096),
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
		# x = self.features(x)
		x = self.avgpool(x)
		x = torch.flatten(x, 1)
		x = self.regressor(x)
		return x

	def training_step(self, batch: tuple, batch_idx: int) -> STEP_OUTPUT:
		x, y = batch
		y_hat = self(x)
		loss = torch.sqrt(F.mse_loss(y_hat, y))
		r2 = r2_score(y_hat.squeeze(), y.squeeze())
		self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
		self.log("train_r2", r2, on_epoch=True, prog_bar=True, logger=True)
		return loss

	def validation_step(self, batch, batch_idx):
		x, y = batch
		y_hat = self(x)
		loss = torch.sqrt(F.mse_loss(y_hat, y))
		r2 = r2_score(y_hat.squeeze(), y.squeeze())
		self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
		self.log("val_r2", r2, on_epoch=True, prog_bar=True, logger=True)

	def test_step(self, batch, batch_idx):
		# this is the test loop
		x, y = batch
		y_hat = self(x)
		loss = torch.sqrt(F.mse_loss(y_hat, y))
		r2 = r2_score(y_hat.squeeze(), y.squeeze())
		self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
		self.log("val_r2", r2, on_epoch=True, prog_bar=True, logger=True)
		return x, y, y_hat

	def configure_optimizers(self) -> Any:
		optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
		lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
			optimizer,
			T_max=100,
			eta_min=1e-6,
		)
		return {
			"optimizer": optimizer,
			"lr_scheduler": lr_scheduler,
			"monitor": "val_loss",
		}


def run_predict(koolset: Koolset, model: pl.LightningModule) -> tuple[list, list]:
	"""
	Run the model on a koolset to generate predictions.

	Parameters
	----------
	koolset : Dataset
		The Dataset to use.
	model : pl.LightningModule

	Returns
	-------
	tuple
		Tuple of the prediction and the associated windows.
	"""
	y_pred = []
	windows = []
	for i in tqdm(range(len(koolset)), desc="Test"):
		x = koolset[i][0]
		y = koolset[i][1]
		w = koolset.get_window(i).tolist()
		windows.append(w)
		y_pred_test = model(x.unsqueeze(0)).detach()[0].tolist()
		y_pred.append(y_pred_test)
	return y_pred, windows


def run_model(win_per_mode: int = 4000, train: bool = False, mode_for_plots: int = 20):
	pl.seed_everything(RANDOM_SEED)
	test_size = 0.2
	val_size = 0.2 / 0.6     # relative to the whole dataset
	np.random.seed(RANDOM_SEED)
	filepath = "cylinder_xi_1_50.h5"

	X, y, w, allmode = data_window_mode(
		filepath=filepath,
		for_rf=False,
		win_per_mode=win_per_mode,
		win_size=(10, 10),
		window_downstream=True,
		mode_collapse=False,
	)
	print(X.shape)
	X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
		X, y, w, test_size=test_size, random_state=RANDOM_SEED
	)
	X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
		X_train, y_train, w_train, test_size=val_size, random_state=RANDOM_SEED
	)

	koolset_train = Koolset(X_train, y_train, w_train)
	koolset_val = Koolset(X_val, y_val, w_val)
	koolset_test = Koolset(X_test, y_test, w_test)
	koolset_predict = Koolset(X_test, y_test, w_test)

	if train:
		model = KoolNet(2 * len(allmode))
	else:
		model = KoolNet.load_from_checkpoint(
			"G:\\PycharmProjects\\ai4er\koolnet\\tb_logs\\lightning_logs\\version_36\\checkpoints\\chk\\epoch=518-val_loss=3.3826.ckpt"
		)

	# Ensure that all operations are deterministic on GPU (if used) for reproducibility
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	precision = "16-mixed"
	torch.set_float32_matmul_precision("medium")

	tensor_logger = TensorBoardLogger("tb_logs")
	checkpoint_callback = ModelCheckpoint(
		monitor="val_loss",
		filename="./chk/{epoch}-{val_loss:.4f}",
		save_top_k=1,  # save the best model
		mode="min",
		every_n_epochs=1
	)

	# early_stop_callback = EarlyStopping(monitor="val_loss", patience=15, mode="min")

	max_epochs = 550
	batch_size = 8
	num_workers = 8
	dm = KoolDataModule(
		batch_size=batch_size,
		koolset_train=koolset_train,
		koolset_val=koolset_val,
		koolset_test=koolset_test,
		koolset_predict=koolset_predict,
	)

	trainer = pl.Trainer(
		max_epochs=max_epochs,
		accelerator="cuda",
		callbacks=[checkpoint_callback, RichProgressBar(refresh_rate=1)],
		precision=precision,
		logger=tensor_logger,
	)

	train_loader = DataLoader(
		koolset_train,
		batch_size=batch_size,
		num_workers=num_workers,
		pin_memory=True,
		shuffle=True,
		persistent_workers=True
	)

	val_loader = DataLoader(
		koolset_val,
		batch_size=batch_size,
		pin_memory=True,
		num_workers=num_workers,
		persistent_workers=True
	)

	test_loader = DataLoader(
		koolset_test,
		batch_size=batch_size,
		pin_memory=True,
		num_workers=num_workers,
		persistent_workers=True
	)

	logger.info("Starting")
	# trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
	trainer.test(model=model, dataloaders=test_loader)

	model.eval()

	y_pred, w_test = run_predict(koolset_test, model)
	y_train_pred, w_train = run_predict(koolset_train, model)

	data, metadata = load_h5(filepath)
	mode = mode_for_plots
	mode_idx = list(metadata["powers"]).index(mode)
	xi = data[mode_idx]
	obst_pos = metadata["obst_x"], metadata["obst_y"], metadata["obst_r"]

	plot_multiple(xi, w_test, obst_pos, y_pred, title="Testing", draw_line=False)

	sns.set_theme()
	plot_pred_obs_dist(obst_pos, w_test, y_pred)
	print("Prediction")
	print(avg_rel_iou(rel_preds=y_pred, obst_pos=obst_pos, win_poss=w_test, filename="pred_iou"))

	print("Training")
	plot_multiple(xi, w_train, obst_pos, y_train_pred, title="Training")
	print(avg_rel_iou(rel_preds=y_train_pred, obst_pos=obst_pos, win_poss=w_train, filename="train_iou"))

	chain_multiple(w_test, model, obst_pos, data, allmode, False)


if __name__ == "__main__":
	run_model()
