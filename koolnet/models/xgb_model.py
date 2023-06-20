import numpy as np
import matplotlib.pyplot as plt

import xgboost as xgb

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score

from koolnet import logger
from koolnet import RANDOM_SEED
from koolnet.data.preprocessing import get_allmode_data
from koolnet.utils.file_ops import load_h5
from koolnet.utils.plotting import plot_multiple


def train_rf(X_train, y_train, n_esti: int = 100):
	rf_model = xgb.XGBRegressor(n_estimators=n_esti, random_state=RANDOM_SEED, tree_method="exact")
	rf_model.fit(X_train, y_train)
	return rf_model


def test_rf(rf_model, X_test, y_test):
	y_pred = rf_model.predict(X_test)
	rmse = np.sqrt(mean_squared_error(y_test, y_pred))
	r2 = r2_score(y_test, y_pred)

	return rmse, r2


def plot_pred_obs_dist(obs, win_coors, pred):
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

	plt.hist(distances)
	plt.show()


def run_rf_plot_pred(win_per_mode):
	test_size = 0.2
	np.random.seed(RANDOM_SEED)
	filepath = "cylinder_xi_1_50.h5"

	X, y, w, _ = get_allmode_data(
		filepath=filepath,
		for_rf=True,
		win_per_mode=win_per_mode,
		win_size=(15, 15),
		window_downstream=True,
		mode_collapse=True
	)
	X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, y, w, test_size=test_size, random_state=RANDOM_SEED)
	rf_model = train_rf(X_train, y_train, n_esti=100)

	rmse, r2 = test_rf(rf_model, X_test, y_test)

	print(f"{rmse = }, {r2 = }")
	y_pred = rf_model.predict(X_test)

	data, metadata = load_h5(filepath)
	mode = 20
	mode_idx = list(metadata["powers"]).index(mode)
	xi = data[mode_idx]
	obst_pos = metadata["obst_x"], metadata["obst_y"], metadata["obst_r"]

	logger.info("Plotting")
	plot_multiple(xi, w_train, obst_pos, y_pred, title="Testing")
	y_train_pred = rf_model.predict(X_train)
	plot_multiple(xi, w_train, obst_pos, y_train_pred, title="Training")
	plot_pred_obs_dist(obst_pos, w_test, y_pred)

	return rmse, r2
	# print(f"{rmse = }\n{r2 = }")


def main():
	run_rf_plot_pred(5000)


if __name__ == "__main__":
	main()
