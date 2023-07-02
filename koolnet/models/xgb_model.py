import numpy as np
import matplotlib.pyplot as plt
import xgboost

import xgboost as xgb
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score

from koolnet import logger
from koolnet import RANDOM_SEED
from koolnet.data.preprocessing import get_allmode_data
from koolnet.utils.file_ops import load_h5
from koolnet.utils.plotting import plot_multiple
from koolnet.utils.metrics import avg_rel_iou

from koolnet.models.predict import chain_mutliple


def train_boost(X_train, y_train, n_esti: int = 100):
	boost_model = xgb.XGBRegressor(n_estimators=n_esti, random_state=RANDOM_SEED, tree_method="exact")
	boost_model.fit(X_train, y_train)
	return boost_model


def test_boost(boost_model, X_test, y_test):
	y_pred = boost_model.predict(X_test)
	rmse = np.sqrt(mean_squared_error(y_test, y_pred))
	r2 = r2_score(y_test, y_pred)

	return rmse, r2


def plot_pred_obs_dist(obs, win_coors, pred) -> None:
	xp, yp, rp = obs
	distances = []
	x, y = [], []
	for w, obst_pos in zip(win_coors, pred):
		wx0, wy0, _, _ = w
		obs_x, obs_y = obst_pos
		xo = wx0 + obs_x
		yo = wy0 + obs_y
		x.append(xo)
		y.append(yo)
		distances.append(np.sqrt((xp - xo)**2 + (yp - yo)**2) / rp)

	f, ax = plt.subplots(figsize=(7, 5))
	sns.despine(f)
	sns.histplot(
		distances,
		edgecolor=".3",
		linewidth=.5,
	)
	plt.savefig("hist_dist_norm_radius.svg", format="svg")
	plt.show()
	f, ax = plt.subplots(figsize=(7, 5))
	sns.despine(f)
	sns.histplot(
		x=(xp - np.array(x)) / (2 * rp),
		y=(yp - np.array(y)) / (2 * rp),
		cbar=True,
		cbar_kws=dict(shrink=.75),
	)
	plt.title("Heatmap of the distance scaled to the diameter of the obstacle")
	plt.savefig("heatmap_dist.svg", format="svg")
	plt.show()


def run_boost_plot_pred(win_per_mode):
	test_size = 0.2
	np.random.seed(RANDOM_SEED)
	filepath = "xi_v3.h5"

	X, y, w, allmode = get_allmode_data(
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
	avg_test_iou = avg_rel_iou(y_pred, obst_pos, w_test, filename="test_iou")
	print(f"{avg_test_iou = }")

	y_train_pred = boost_model.predict(X_train)
	avg_train_iou = avg_rel_iou(y_train_pred, obst_pos, w_train, filename="train_iou")
	print(f"{avg_train_iou = }")
	plot_multiple(xi, w_train, obst_pos, y_train_pred, title="Training")
	plot_pred_obs_dist(obst_pos, w_test, y_pred)

	boost_model.get_booster().feature_names = [f"{t}_{m}" for m in allmode for t in ["real", "abs"]]

	print("figure importance")
	plt.figure(figsize=(20, 20))
	xgboost.plot_importance(boost_model, ax=plt.gca())
	plt.savefig("feature_importance.svg", format="svg")
	plt.show()

	chain_mutliple(w_test, boost_model, obst_pos, data, allmode, True)

	return rmse, r2
	# print(f"{rmse = }\n{r2 = }")


def main():
	run_boost_plot_pred(2000)


if __name__ == "__main__":
	main()
