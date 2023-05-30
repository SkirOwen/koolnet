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


def bar(avg, win_per_mode):
	from tqdm import tqdm
	rmse_lst, r2_lst = [], []
	for i in tqdm(range(avg)):
		rmse, r2 = foo(win_per_mode)
		rmse_lst.append(rmse)
		r2_lst.append(r2)
	rmse_m = np.mean(rmse_lst)
	r2_m = np.mean(r2_lst)
	print(f"{rmse_m = }\n{r2_m = }")


def foo(win_per_mode):
	global random_seed
	random_seed = 17
	test_size = 0.2

	X, y = get_allmode_data(for_rf=True, win_per_mode=win_per_mode)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)
	rf_model = train_rf(X_train, y_train, n_esti=100)
	rmse, r2 = test_rf(rf_model, X_test, y_test)

	return rmse, r2
	# print(f"{rmse = }\n{r2 = }")


def main():
	bar(50, 1000)


if __name__ == "__main__":
	main()
