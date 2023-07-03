from __future__ import annotations

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import seaborn as sns

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import ModelSummary, RichModelSummary
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import RichProgressBar, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger

from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchmetrics.functional import r2_score

from sklearn.model_selection import train_test_split

from tqdm import tqdm

from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT

from koolnet import logger
from koolnet import RANDOM_SEED
from koolnet.data.preprocessing import data_window_mode
from koolnet.data.windows import get_data_window, window_coord_centre_point
from koolnet.dataloading.koolload import KoolDataModule
from koolnet.dataset.koolset import Koolset
from koolnet.utils import load_h5
from koolnet.utils.metrics import avg_rel_iou, rel_iou
from koolnet.utils.plotting import plot_multiple, plot_pred_obs_dist
from koolnet.models.predict import chain_mutliple


class KoolNet(pl.LightningModule):
	def __init__(self, mode_nbr: int, dropout: float = 0.17, lr: float = 1e-5):
		super().__init__()
		self.save_hyperparameters()
		self.mode_nbr = mode_nbr
		self.lr = lr

		# self.features = nn.Sequential(
		# 	nn.Conv2d(self.mode_nbr, 64, kernel_size=3, stride=1, padding=2),
		# 	nn.Tanh(),
		# 	nn.MaxPool2d(kernel_size=3, stride=2),
		#
		# 	nn.Conv2d(64, 192, kernel_size=5, padding=2),
		# 	nn.Tanh(),
		# 	nn.MaxPool2d(kernel_size=3, stride=2),
		#
		# 	nn.Conv2d(192, 384, kernel_size=3, padding=1),
		# 	nn.Tanh(),
		#
		# 	nn.Conv2d(384, 256, kernel_size=3, padding=1),
		# 	nn.Tanh(),
		#
		# 	nn.Conv2d(256, 256, kernel_size=3, padding=1),
		# 	nn.Tanh(),
		# 	nn.MaxPool2d(kernel_size=3, stride=2),
		# )

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
			nn.MaxPool2d(
				kernel_size=3,
				# stride=2,
			),
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
			T_max=100,# 500 # 100
			eta_min=1e-6    # 1e-7
		)
		return {
			"optimizer": optimizer,
			"lr_scheduler": lr_scheduler,
			"monitor": "val_loss",
		}


def dist_win_obst(obst_xy: tuple[int, int], win_coords: tuple[int, int, int, int]) -> tuple[int, int]:
	x0, y0, _, _ = win_coords
	dx = obst_xy[0] - x0
	dy = obst_xy[1] - y0
	return dx, dy


def chain_predict_one(window_coor, model, obst_pos, data, allmodes, for_rf):
	"""if output 0 converges right away
	-1 out-of-bound right away
	negative out-of-bound in n step
	postive converges in n step
	"""
	koopmodes_xy = data.shape[1::]
	obst_xy = obst_pos[:2]

	dist_x, dist_y = dist_win_obst(obst_xy, window_coor)
	y = [dist_x, dist_y]
	x_mode_r = []
	x_mode_abs = []
	x_data = []

	for mode in allmodes:
		# koopmode = get_koop_mode(data, mode)
		# TODO: fix this!
		mode_idx = list(allmodes).index(mode)
		koopmode = data[mode_idx]

		data_window = get_data_window(koopmode, window_coor)
		wind_r, wind_abs = np.real(data_window), np.abs(data_window)
		x_mode_r.append(wind_r)
		x_mode_abs.append(wind_abs)
	x_data.append((*x_mode_r, *x_mode_abs))
	x_data = np.array(x_data)

	div_count = 0
	score = 0
	k = 0
	while score <= 0:
		x = torch.Tensor(x_data)
		y = model(x).detach()[0].tolist()
		score = rel_iou(y, obst_pos, window_coor)
		if score > 0:
			div_count += 1
			break
		true_y = (window_coor[0] + y[0]), (window_coor[1] + y[1])
		if not (5 <= true_y[0] <= koopmodes_xy[0] - 5) or not (5 <= true_y[1] <= koopmodes_xy[1] - 5):
			# print("Ha")
			div_count = -div_count
			break
		k += 1
		if k > 100:
			break
		div_count += 1
		window_coor = window_coord_centre_point(
			true_y,
			win_size=(10, 10),
		)
		x_mode_r = []
		x_mode_abs = []
		x_data = []

		for mode in allmodes:
			# koopmode = get_koop_mode(data, mode)
			# TODO: fix this!
			mode_idx = list(allmodes).index(mode)
			koopmode = data[mode_idx]

			data_window = get_data_window(koopmode, window_coor)
			wind_r, wind_abs = np.real(data_window), np.abs(data_window)
			x_mode_r.append(wind_r)
			x_mode_abs.append(wind_abs)
		x_data.append((*x_mode_r, *x_mode_abs))
		x_data = np.array(x_data)

	return div_count - 1


def chain_mutliple(windows, model, obst_pos, data, allmodes, for_rf):
	rslt = []
	for w in tqdm(windows):
		div_count = chain_predict_one(w, model, obst_pos, data, allmodes, for_rf)
		rslt.append(div_count)

	labels, counts = np.unique(rslt, return_counts=True)
	plt.bar(labels, counts, align="center")
	plt.ylabel("Counts")
	plt.xlabel("Chain prediction convergence")
	plt.savefig("Chain_pred.svg", format="svg")
	plt.show()
	return rslt


def run_model():
	pl.seed_everything(RANDOM_SEED)
	win_per_mode = 4000
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

	# model = KoolNet(2 * len(allmode))
	# model(X_train[0])
	model = KoolNet.load_from_checkpoint("G:\\PycharmProjects\\ai4er\koolnet\\tb_logs\\lightning_logs\\version_36\\checkpoints\\chk\\epoch=518-val_loss=3.3826.ckpt")

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
	y_pred = []
	w_test = []
	for i in tqdm(range(len(koolset_test)), desc="Test"):
		x = koolset_test[i][0]
		y = koolset_test[i][1]
		w = koolset_test.get_window(i).tolist()
		w_test.append(w)
		y_pred_test = model(x.unsqueeze(0)).detach()[0].tolist()
		y_pred.append(y_pred_test)
	#

	y_train_pred = []
	w_train = []
	for i in tqdm(range(len(koolset_train)), desc="Test"):
		x = koolset_train[i][0]
		y = koolset_train[i][1]
		w = koolset_train.get_window(i).tolist()
		w_train.append(w)
		y_train_ = model(x.unsqueeze(0)).detach()[0].tolist()
		y_train_pred.append(y_train_)

	data, metadata = load_h5(filepath)
	mode = 20
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

	chain_mutliple(w_test, model, obst_pos, data, allmode, False)


if __name__ == "__main__":
	run_model()
