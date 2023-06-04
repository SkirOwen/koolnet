import numpy as np

from tqdm import tqdm

from koolnet.utils.file_ops import load_h5
from koolnet.data.windows import get_data_win_size, get_data_window, gen_window_coord


def dist_win_obst(obst_xy: tuple[int, int], win_coords: tuple[int, int, int, int]) -> tuple[int, int]:
	x0, y0, _, _ = win_coords
	dx = x0 - obst_xy[0]
	dy = y0 - obst_xy[1]
	return dx, dy


def get_koop_mode(data, mode: int = 1):
	# TODO: this is currently broken, as I need to save the Idd or the xi_
	# TODO: and not only xi
	xi = data[mode - 1]
	xi_ = (-1j * xi.reshape(100, 400)).T
	return xi_


def get_mode_data(mode: int):
	# TODO: this is currently broken, as I need to save the Idd or the xi_
	# TODO: and not only xi
	data, metadata = load_h5("xi.h5")
	koopmode = get_koop_mode(data, mode)
	obst_pos = metadata["obst_x"], metadata["obst_y"], metadata["obst_r"]
	obst_xy = obst_pos[:2]

	data_window = get_data_win_size(koopmode, win_size=(10, 10), obst_pos=obst_pos)

	return data_window, obst_xy


def get_allmode_data(for_rf: bool, win_per_mode: int, win_size: tuple):
	# TODO: Careful this is for one file, of one simulation
	# TODO: having a way to see the sampled windows
	data, metadata = load_h5("xi.h5")
	allmodes = metadata["powers"]
	xi = data.reshape(-1, 100, 400)

	obst_pos = metadata["obst_x"], metadata["obst_y"], metadata["obst_r"]
	obst_xy = obst_pos[:2]
	# TODO: fix the position of the obstacle

	x_data = []
	y_data = []
	win_data = []

	for w in tqdm(range(win_per_mode), leave=False, desc="Wind #"):
		koopmodes_xy = xi.shape[1::-1]
		# TODO: data is not the correct shape
		window_coords = gen_window_coord(koopmodes_xy, win_size=win_size,  obst_pos=obst_pos)
		win_data.append(window_coords)
		wx0, wy0, _, _ = window_coords

		# TODO: check this
		dist_x, dist_y = dist_win_obst(obst_xy, window_coords)

		y_data.append((dist_x, dist_y))
		x_mode_r = []
		x_mode_abs = []
		for mode in tqdm(allmodes, leave=False, desc="Mode #"):
			koopmode = get_koop_mode(data, mode)
			data_window = get_data_window(koopmode, window_coords)
			if for_rf:
				wind_r, wind_abs = np.real(data_window), np.abs(data_window)
				x_mode_r.append(wind_r.flatten().sum())
				x_mode_abs.append(wind_abs.flatten().sum())
		if for_rf:
			x_data.append((np.sum(x_mode_r), np.sum(x_mode_abs)))

	return np.array(x_data), np.array(y_data), np.array(win_data)

