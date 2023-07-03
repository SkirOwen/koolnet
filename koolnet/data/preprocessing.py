import numpy as np

from tqdm import tqdm

from typing import Iterable

from koolnet.utils.file_ops import load_h5
from koolnet.data.windows import get_data_win_size, get_data_window, gen_window_coord


def dist_win_obst(obst_xy: tuple[int, int], win_coords: tuple[int, int, int, int]) -> tuple[int, int]:
	"""
	Calculate the relative postion of the obstacle to the window.

	Parameters
	----------
	obst_xy : tuple of 2 ints
		Tuple (x, y) of the position of the obstacle.
	win_coords : tuples of 4 ints
		Tuple of the opposite corner of the window.
		Should be ordered as lowest (x, y) corner followed by the opposite, as:
		wx0, wy0, wx1, wy1

	Returns
	-------
	tuple of 2 ints
		Tuple of the (dx, dy) coordinates of the obstacle relative to the window.
	"""
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


def get_data_modes(allmodes: Iterable, data: np.ndarray, window_coords: tuple, for_rf: bool) -> tuple[list, list]:
	"""
	Get the data in a window across multiple modes.
	The data given per mode is a tuple of the real and abs.
	If it is for a decision tree, set for_rf to True, this will sum all the
	values in a window to one value.

	Parameters
	----------
	allmodes : Iterable
		An iterable of all the modes to use.
	data : ndarray
		2D ndarray of the data the to slice
	window_coords : tuples of 4 ints
		Tuple of the opposite corner of the window.
		Should be ordered as lowest (x, y) corner followed by the opposite, as:
		wx0, wy0, wx1, wy1
	for_rf : bool
		If False, return the window as is.
		Otherwise, flattens it and sums it to get one number per window,
		to be compatible with decision trees.

	Returns
	-------
	tuple of list
		A list of the window for the real part, and one for the abs.

	"""
	x_mode_r = []
	x_mode_abs = []
	for mode in allmodes:
		# koopmode = get_koop_mode(data, mode)
		# TODO: fix this!
		mode_idx = list(allmodes).index(mode)
		koopmode = data[mode_idx]

		data_window = get_data_window(koopmode, window_coords)
		wind_r, wind_abs = np.real(data_window), np.abs(data_window)
		if for_rf:
			x_mode_r.append(wind_r.flatten().sum())
			x_mode_abs.append(wind_abs.flatten().sum())
		else:
			x_mode_r.append(wind_r)
			x_mode_abs.append(wind_abs)
	return x_mode_r, x_mode_abs


def data_window_mode(
		filepath: str,
		for_rf: bool,
		win_per_mode: int,
		win_size: tuple[int, int],
		window_downstream: bool,
		mode_collapse: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""
	Generate the data for multiple windows.
	Per window, extracts the data across all the modes present in the file.
	Returns the data extracted, the relative position of the obstacle to the window,
	the coordinates of the windows used, and the array of all the modes present/used.

	Parameters
	----------
	filepath : str
		Filepath to the file to use. Needs to be an h5 file.
	for_rf : bool
		If False, return the window as is.
		Otherwise, flattens it and sums it to get one number per window,
		to be compatible with decision trees.
	win_per_mode : int
		Number of windows to use.
	win_size : tuple of 2 ints
		Tuple (l, w) to give the length (l) and width (w) of the window to generate.
	window_downstream : bool
		If True, only samples from x bigger than the x position of the obstacle
		plus three times its radius, i.e. (x >= 3 * radius + obst_pos_x). The default is False.
	mode_collapse : bool, optional
		If True collapse sum across all the modes.

	Returns
	-------
	tuple of ndarray
		Corresponding to:
			- data
			- relative coordinates of the obstacle to the window
			- the windows coordinates used
			- array all the nodes used

	See Also
	--------
	gen_window_coord: for more details on the window creation.
	"""
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

		x_mode_r, x_mode_abs = get_data_modes(allmodes, data, window_coords, for_rf)

		if mode_collapse:
			x_data.append((np.sum(x_mode_r), np.sum(x_mode_abs)))
		else:
			x_data.append((*x_mode_r, *x_mode_abs))

	return np.array(x_data), np.array(y_data), np.array(win_data), allmodes
