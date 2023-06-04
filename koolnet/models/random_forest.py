import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score

from koolnet.data.preprocessing import get_allmode_data


def train_rf(X_train, y_train, n_esti: int = 100):
	rf_model = RandomForestRegressor(n_estimators=n_esti, random_state=random_seed)
	rf_model.fit(X_train, y_train)
	return rf_model


def test_rf(rf_model, X_test, y_test):
	y_pred = rf_model.predict(X_test)
	rmse = np.sqrt(mean_squared_error(y_test, y_pred))
	r2 = r2_score(y_test, y_pred)

	return rmse, r2


def cal_iou(pred, obst_pos) -> float:
	# assuming same radius
	xp, yp, rp = pred
	xo, yo, ro = obst_pos
	area_p = np.pi * rp ** 2
	area_o = np.pi * ro ** 2

	d = np.sqrt((xp - xo)**2 + (yp - yo)**2)

	if d >= rp + ro:
		iou = 0
	elif d <= np.abs(rp - ro):
		iou = 1
	else:
		ai = (
			rp**2 * np.arccos((d**2 + rp**2 - ro**2) / (2 * d * rp)) +
			ro**2 * np.arccos((d**2 + ro**2 - rp**2) / (2 * d * ro)) -
			0.5 * np.sqrt((-d + rp + ro) * (d + rp - ro) * (d - rp + ro) * (d + rp + ro))
		)
		au = area_p + area_o - ai

		iou = ai / au

	return iou


def bar(avg, win_per_mode):
	from tqdm import tqdm
	rmse_lst, r2_lst = [], []
	for i in tqdm(range(avg), leave=False):
		rmse, r2 = foo(win_per_mode)
		rmse_lst.append(rmse)
		r2_lst.append(r2)
	rmse_m = np.mean(rmse_lst)
	r2_m = np.mean(r2_lst)
	print(f"{rmse_m = }\n{r2_m = }")

# TODO: add an IoU metric


def foo(win_per_mode):
	global random_seed
	win_per_mode = 1000
	random_seed = 17
	test_size = 0.2

	X, y, w = get_allmode_data(for_rf=True, win_per_mode=win_per_mode, win_size=(10, 10))
	X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, y, w, test_size=test_size, random_state=random_seed)
	rf_model = train_rf(X_train, y_train, n_esti=100)
	rmse, r2 = test_rf(rf_model, X_test, y_test)

	return rmse, r2
	# print(f"{rmse = }\n{r2 = }")


def main():
	bar(10, 8000)


if __name__ == "__main__":
	main()
