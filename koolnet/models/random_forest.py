from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
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
from koolnet.utils.plotting import plot_multiple, plot_pred_obs_dist
from koolnet.utils.metrics import avg_rel_iou
from koolnet.models.predict import chain_multiple


def train_rf(X_train, y_train, n_esti: int = 210):
	# RandomForestRegressor(max_depth=10, max_features='log2', n_estimators=200)
	rf_model = RandomForestRegressor(
		n_estimators=n_esti,
		random_state=RANDOM_SEED,
		max_features=None,
		max_depth=90,
		n_jobs=4,
		verbose=0,
	)
	# rf_model = RandomForestRegressor(
	# 	n_estimators=100,
	# 	random_state=RANDOM_SEED,
	# 	n_jobs=4,
	# 	verbose=0,
	# )
	rf_model.fit(X_train, y_train)
	return rf_model


def test_rf(rf_model, X_test, y_test):
	y_pred = rf_model.predict(X_test)
	rmse = np.sqrt(mean_squared_error(y_test, y_pred))
	r2 = r2_score(y_test, y_pred)

	return rmse, r2


def hyper_param(X_train, y_train, jobs: int = -1):
	from sklearn.model_selection import GridSearchCV

	max_depth = [x for x in range(10, 120, 10)]
	max_depth.append(None)

	param_grid = {
		'n_estimators': [x for x in range(10, 500, 20)],
		'max_features': ['sqrt', 'log2', None],
		'max_depth': max_depth,
		'min_samples_split': [2, 5, 10],
		'min_samples_leaf': [1, 2, 4],
		'bootstrap': [True, False]
	}

	grid_search = GridSearchCV(RandomForestRegressor(), param_grid=param_grid, verbose=10, n_jobs=jobs)
	grid_search.fit(X_train, y_train)
	# RandomForestRegressor(max_depth=10, max_features='log2', n_estimators=200)
	# RandomForestRegressor(max_depth=90, max_features=None, n_estimators=210)
	print(grid_search.best_estimator_)


def run_rf_plot_pred(win_per_mode: int = 2000, mode_for_plots: int = 20):
	console = Console()
	test_size = 0.2
	np.random.seed(RANDOM_SEED)
	filepath = "cylinder_xi_1_50.h5"

	X, y, w, allmode = data_window_mode(
		filepath=filepath,
		for_rf=True,
		win_per_mode=win_per_mode,
		win_size=(10, 10),
		window_downstream=True,
		mode_collapse=False,
	)
	X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, y, w, test_size=test_size, random_state=RANDOM_SEED)
	rf_model = train_rf(X_train, y_train, n_esti=210)
	rmse, r2 = test_rf(rf_model, X_test, y_test)

	console.print(
		Panel(
			f"{rmse = }\n"
			f"{r2   = }",
			title="Results of the Random Forest"
		)
	)
	y_pred = rf_model.predict(X_test)

	data, metadata = load_h5(filepath)
	mode_idx = list(metadata["powers"]).index(mode_for_plots)
	xi = data[mode_idx]
	obst_pos = metadata["obst_x"], metadata["obst_y"], metadata["obst_r"]

	# Plotting on Flow
	logger.info("Plotting")
	# Test
	plot_multiple(xi, w_test, obst_pos, y_pred, title="Testing", draw_line=False, model_name="RF")
	# Train
	y_train_pred = rf_model.predict(X_train)
	plot_multiple(xi, w_train, obst_pos, y_train_pred, title="Training", model_name="RF")

	sns.set_theme()
	plot_pred_obs_dist(obst_pos, w_test, y_pred)

	# Feature importance (impurity)
	feature_name = [f"{t}_{m}" for m in allmode for t in ["real", "abs"]]
	importance = rf_model.feature_importances_
	std = np.std([tree.feature_importances_ for tree in rf_model.estimators_], axis=0)
	forest_importances = pd.Series(importance, index=feature_name)
	fig, ax = plt.subplots(figsize=(15, 10))
	forest_importances.plot.bar(yerr=std, ax=ax)
	ax.set_title("Feature importances using MDI")
	ax.set_ylabel("Mean decrease in impurity")
	fig.tight_layout()
	plt.savefig("Impurity_feature_importance.svg", format="svg")
	plt.show()

	# Permutation importance
	from sklearn.inspection import permutation_importance
	result = permutation_importance(
		rf_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
	)
	forest_importances = pd.Series(result.importances_mean, index=feature_name)
	fig, ax = plt.subplots(figsize=(15, 10))
	forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
	ax.set_title("Feature importances using permutation on full model")
	ax.set_ylabel("Mean accuracy decrease")
	fig.tight_layout()
	plt.savefig("all_model_feature_importance.svg", format="svg")
	plt.show()

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

	# Chain
	chain_multiple(w_test, rf_model, obst_pos, data, allmode, True)
	return rmse, r2
	# print(f"{rmse = }\n{r2 = }")


def main():
	run_rf_plot_pred(2000)


if __name__ == "__main__":
	main()
