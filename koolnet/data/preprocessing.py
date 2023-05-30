import numpy as np

from tqdm import tqdm

from koolnet.utils.file_ops import load_h5
from koolnet.data.windows import get_data_win_size,get_data_window, gen_window_coord


def get_koop_mode(data, mode: int = 1):
	# TODO: this is currently broken, as I need to save the Idd or the xi_
	# TODO: and not only xi
	xi = data[mode - 1]
	xi_ = (-1j * xi.reshape(100, 400)).T
	return xi_


def get_mode_data(mode: int):
	# TODO: this is currently broken, as I need to save the Idd or the xi_
	# TODO: and not only xi
	data, metadata = load_h5("xi")
	koopmode = get_koop_mode(data, mode)
	obst_pos = metadata["obst_x"], metadata["obst_y"], metadata["obst_r"]
	obst_xy = obst_pos[:2]

	data_window = get_data_win_size(koopmode, win_size=(10, 10), obst_pos=obst_pos)

	return data_window, obst_xy


def get_allmode_data(for_rf: bool, win_per_mode: int):
	# TODO: Careful this is for one file, of one simulation
	# TODO: having a way to see the sampled windows
	data, metadata = load_h5("xi")
	allmodes = metadata["powers"]

	obst_pos = metadata["obst_x"], metadata["obst_y"], metadata["obst_r"]
	obst_xy = obst_pos[:2]

	x = []
	y = []

	for mode in tqdm(allmodes):
		for w in range(win_per_mode):
			koopmode = get_koop_mode(data, mode)
			koopmodes_xy = koopmode.shape[:2]
			window_coords = gen_window_coord(koopmodes_xy, win_size=(10, 10),  obst_pos=obst_pos)
			data_window = get_data_window(koopmode, window_coords)
			wx0, _, wy0, _ = window_coords

			dist_x = wx0 - metadata["obst_x"]
			dist_y = wy0 - metadata["obst_y"]
			y.append((dist_x, dist_y))
			if for_rf:
				wind_r, wind_c = np.real(data_window), np.imag(data_window)
				x.append((wind_r.flatten().sum(), wind_c.flatten().sum()))
			else:
				x.append(data_window)

	return np.array(x), np.array(y)


####
# 100, real_only
# rmse
# Out[63]: 56.10645682862513
# r2 = r2_score(y_test, y_pred)
# r2
# Out[65]: -0.005008137710641458
#
# 300, real_only
# rmse
# Out[73]: 61.07491857575466
# r2
# Out[74]: -0.009413718719932684
#
# 300, real and imag sum
# rmse
# Out[78]: 50.265797687028005
# r2
# Out[79]: 0.3665034130767686
