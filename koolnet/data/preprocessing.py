import numpy as np

from tqdm import tqdm

from koolnet.utils.file_ops import load_h5
from koolnet.data.windows import get_data_win_size, get_data_window, gen_window_coord


def dist_win_obst(obst_xy: tuple[int, int], win_coords: tuple[int, int, int, int]) -> tuple[int, int]:
	x0, y0, _, _ = win_coords
	dx = obst_xy[0] - x0
	dy = obst_xy[1] - y0
	return dx, dy


def get_koop_mode(data, mode: int = 1):
	# TODO: this is currently broken, as I need to save the Idd or the xi_
	# TODO: and not only xi
	xi = data[mode - 1]
	xi_ = (-1j * xi.reshape(100, 400)).T
	return xi_


def get_mode_data(filepath: str, mode: int) -> tuple:
	# TODO: this is currently broken, as I need to save the Idd or the xi_
	# TODO: and not only xi
	data, metadata = load_h5(filepath)
	# koopmode = get_koop_mode(data, mode)
	mode_idx = list(metadata["powers"]).index(mode)
	koopmode = data[mode_idx]
	obst_pos = metadata["obst_x"], metadata["obst_y"], metadata["obst_r"]
	obst_xy = obst_pos[:2]

	data_window = get_data_win_size(koopmode, win_size=(10, 10), obst_pos=obst_pos)

	return data_window, obst_xy


def get_allmode_data(
		filepath: str,
		for_rf: bool,
		win_per_mode: int,
		win_size: tuple,
		window_downstream: bool,
		mode_collapse: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	# TODO: Careful this is for one file, of one simulation
	# TODO: having a way to see the sampled windows
	data, metadata = load_h5(filepath)
	allmodes = metadata["powers"]

	obst_pos = metadata["obst_x"], metadata["obst_y"], metadata["obst_r"]
	obst_xy = obst_pos[:2]
	# TODO: fix the position of the obstacle

	x_data = []
	y_data = []
	win_data = []

	for w in tqdm(range(win_per_mode), leave=False, desc="Wind #"):
		koopmodes_xy = data.shape[1::]

		window_coords = gen_window_coord(
			koopmodes_xy,
			win_size=win_size,
			obst_pos=obst_pos,
			downstream=window_downstream,
		)
		win_data.append(window_coords)
		wx0, wy0, _, _ = window_coords

		dist_x, dist_y = dist_win_obst(obst_xy, window_coords)

		y_data.append((dist_x, dist_y))
		x_mode_r = []
		x_mode_abs = []
		for mode in tqdm(allmodes, leave=False, desc="Mode #"):
			# koopmode = get_koop_mode(data, mode)
			# TODO: fix this!
			mode_idx = list(metadata["powers"]).index(mode)
			koopmode = data[mode_idx]

			data_window = get_data_window(koopmode, window_coords)
			if for_rf:
				wind_r, wind_abs = np.real(data_window), np.abs(data_window)
				x_mode_r.append(wind_r.flatten().sum())
				x_mode_abs.append(wind_abs.flatten().sum())

		if mode_collapse:
			x_data.append((*x_mode_r, *x_mode_abs))
		else:
			x_data.append((np.sum(x_mode_r), np.sum(x_mode_abs)))

	return np.array(x_data), np.array(y_data), np.array(win_data), allmodes
