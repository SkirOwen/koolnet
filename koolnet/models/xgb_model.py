from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn
import xgboost as xgb

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from koolnet import logger
from koolnet import RANDOM_SEED
from koolnet.data.preprocessing import data_window_mode
from koolnet.utils import load_h5
from koolnet.utils.metrics import avg_rel_iou
from koolnet.utils.plotting import plot_multiple
from koolnet.utils.plotting import plot_pred_obs_dist
from koolnet.models.predict import chain_mutliple


def train_boost(X_train: np.ndarray, y_train: np.ndarray, n_esti: int = 100) -> sklearn.pipeline.Pipelin:
	boost_model = xgb.XGBRegressor(n_estimators=n_esti, random_state=RANDOM_SEED, tree_method="exact")
	boost_model.fit(X_train, y_train)
	return boost_model


def test_boost(boost_model: sklearn.pipeline.Pipelin, X_test: np.ndarray, y_test: np.ndarray) -> tuple[float, float]:
	y_pred = boost_model.predict(X_test)
	rmse = np.sqrt(mean_squared_error(y_test, y_pred))
	r2 = r2_score(y_test, y_pred)
	return rmse, r2


def run_boost_plot_pred(win_per_mode: int) -> None:
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

	print(f"{rmse = }, {r2 = }")
	y_pred = boost_model.predict(X_test)

	data, metadata = load_h5(filepath)
	mode = 20
	mode_idx = list(metadata["powers"]).index(mode)
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
	avg_test_iou = avg_rel_iou(y_pred, obst_pos, w_test, filename="test_iou")
	print(f"{avg_test_iou = }")
	avg_train_iou = avg_rel_iou(y_train_pred, obst_pos, w_train, filename="train_iou")
	print(f"{avg_train_iou = }")

	chain_mutliple(w_test, boost_model, obst_pos, data, allmode, True)



def main():
	run_boost_plot_pred(2000)


if __name__ == "__main__":
	main()
