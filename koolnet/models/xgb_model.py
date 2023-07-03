from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn
import xgboost as xgb

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from koolnet import logger
from koolnet import RANDOM_SEED
from koolnet.data.preprocessing import data_window_mode
from koolnet.utils import load_h5
from koolnet.utils.metrics import avg_rel_iou
from koolnet.utils.plotting import plot_multiple
from koolnet.utils.plotting import plot_pred_obs_dist
from koolnet.models.predict import chain_multiple


def train_boost(X_train: np.ndarray, y_train: np.ndarray, n_esti: int = 100) -> sklearn.pipeline.Pipelin:
	boost_model = xgb.XGBRegressor(n_estimators=n_esti, random_state=RANDOM_SEED, tree_method="exact")
	boost_model.fit(X_train, y_train)
	return boost_model


def test_boost(boost_model: sklearn.pipeline.Pipelin, X_test: np.ndarray, y_test: np.ndarray) -> tuple[float, float]:
	y_pred = boost_model.predict(X_test)
	rmse = np.sqrt(mean_squared_error(y_test, y_pred))
	r2 = r2_score(y_test, y_pred)
	return rmse, r2


def run_boost_plot_pred(win_per_mode: int = 2000, mode_for_plots: int = 20) -> None:
	console = Console()
	test_size = 0.2
	np.random.seed(RANDOM_SEED)
	filepath = "xi_v3.h5"

	X, y, w, allmode = data_window_mode(
		filepath=filepath,
		for_rf=True,
		win_per_mode=win_per_mode,
		win_size=(10, 10),
		window_downstream=True,
		mode_collapse=False
	)
	X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, y, w, test_size=test_size, random_state=RANDOM_SEED)
	boost_model = train_boost(X_train, y_train, n_esti=100)

	rmse, r2 = test_boost(boost_model, X_test, y_test)

	console.print(
		Panel(
			f"{rmse = }\n"
			f"{r2   = }",
			title="Results of the XGBoost"
		)
	)
	y_pred = boost_model.predict(X_test)

	data, metadata = load_h5(filepath)
	mode_idx = list(metadata["powers"]).index(mode_for_plots)
	xi = data[mode_idx]
	obst_pos = metadata["obst_x"], metadata["obst_y"], metadata["obst_r"]

	logger.info("Plotting")
	plot_multiple(xi, w_test, obst_pos, y_pred, title="Testing")

	y_train_pred = boost_model.predict(X_train)
	plot_multiple(xi, w_train, obst_pos, y_train_pred, title="Training")
	plot_pred_obs_dist(obst_pos, w_test, y_pred)

	boost_model.get_booster().feature_names = [f"{t}_{m}" for m in allmode for t in ["real", "abs"]]

	print("figure importance")
	plt.figure(figsize=(20, 20))
	xgb.plot_importance(boost_model, ax=plt.gca())
	plt.savefig("feature_importance.svg", format="svg")
	plt.show()

	sns.set_theme()

	# IoU
	pred_lst_iou = avg_rel_iou(rel_preds=y_pred, obst_pos=obst_pos, win_poss=w_test, filename="pred_iou")
	train_lst_iou = avg_rel_iou(rel_preds=y_train_pred, obst_pos=obst_pos, win_poss=w_train, filename="train_iou")

	pred_avg_iou = sum(pred_lst_iou) / len(pred_lst_iou)
	train_avg_iou = sum(train_lst_iou) / len(train_lst_iou)

	table_iou = Table(title="IoU", box=box.MINIMAL)
	table_iou.add_column("Type", justify="left", no_wrap=True)
	table_iou.add_column("Prediction", justify="right", no_wrap=True)
	table_iou.add_column("Training", justify="right", no_wrap=True)

	pred_z_iou = len([i for i in pred_lst_iou if i == 0])
	pred_nz_iou = len([i for i in pred_lst_iou if i != 0])

	train_z_iou = len([i for i in train_lst_iou if i == 0])
	train_nz_iou = len([i for i in train_lst_iou if i != 0])

	table_iou.add_row("Non-Zero IoU", str(pred_nz_iou), str(train_nz_iou))
	table_iou.add_row("Zero IoU", str(pred_z_iou), str(train_z_iou))
	table_iou.add_row("Average", str(pred_avg_iou), str(train_avg_iou))

	console.print(table_iou)

	chain_multiple(w_test, boost_model, obst_pos, data, allmode, True)


def main():
	run_boost_plot_pred(2000)


if __name__ == "__main__":
	main()
