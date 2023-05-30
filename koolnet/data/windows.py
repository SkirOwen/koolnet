import numpy as np



def get_window_coord(koop_modes_xy: tuple[int, int], win_size: tuple[int, int], obst_pos: tuple) -> nd.array:
	# pick a random points
	# check if in bounds and the cylinder is not in
	# create a padded area of padding == win_size, so guarantee to be in bounds.

	x, y = koop_modes_xy
	obst_x, obst_y, obst_r = obst_pos

	win_size_x, win_size_y = win_size

	wx0 = np.random.randint(obst_x + (2 * obst_r), x - win_size_x)
	wy0 = np.random.randint(obst_y + (2 * obst_r), y - win_size_y)

	wx1 = wx0 + win_size_x
	wy1 = wy0 + win_size_y

	return wx0, wx1, wy0, wy1


def get_data_window(koop_modes: np.ndarray, window_coords: tuple[int, int, int, int]) -> np.ndarray:
	wx0, wx1, wy0, wy1 = window_coords
	# TODO: check the off by one for the slicing
	# np.random.randint is [a, b) is semi-open
	return koop_modes[wx0:wx1, wy0:wy1]


def get_window_and_data(koop_modes: np.ndarray, win_size: tuple[int, int], obst_pos: tuple) -> None:
	koop_modes_xy = koop_modes.shape[:1]
	window_coords = get_window_coord(koop_modes_xy, win_size, obst_pos)

	data_window = get_data_window(koop_modes, window_coords)
