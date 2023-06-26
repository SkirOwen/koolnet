from __future__ import annotations

import numpy as np


def gen_window_coord(
		koop_modes_xy: tuple[int, int],
		win_size: tuple[int, int],
		obst_pos: tuple,
		downstream: bool = False,
		padding: int = 10,
) -> tuple:
	# pick a random points
	# check if in bounds and the cylinder is not in
	# create a padded area of padding == win_size, so guarantee to be in bounds.
	# TODO: set a seed for the random.randint

	x, y = koop_modes_xy
	obst_x, obst_y, obst_r = obst_pos

	win_size_x, win_size_y = win_size

	# TODO: check if win_size > size and size - obst, otherwise goes to default values
	# if win_size_x >= x or win_size_y >= y:
	# 	raise ValueError("win_size is bigger than the available size")

	# TODO: only sample downstream, though I would need to assume max(x, y) is the downstream.
	min_x = padding
	min_y = padding
	if downstream:
		# Assuming downstream is x increasing.
		# padding the obstacle with an obst_r
		min_x = obst_x + 2 * obst_r

	wx0 = np.random.randint(min_x, x - win_size_x - padding)
	wy0 = np.random.randint(min_y, y - win_size_y - padding)

	wx1 = wx0 + win_size_x
	wy1 = wy0 + win_size_y

	if not downstream:
		while (wx0 <= obst_x + (2 * obst_r) < wx1) and (wy0 <= obst_y + (2 * obst_r) < wy1):
			# TODO: need to check the other corner
			# TODO: there must be a better way
			wx0 = np.random.randint(0, x - win_size_x - padding)
			wy0 = np.random.randint(0, y - win_size_y - padding)

			wx1 = wx0 + win_size_x
			wy1 = wy0 + win_size_y

	return wx0, wy0, wx1, wy1


def get_data_window(koop_modes: np.ndarray, window_coords: tuple[int, int, int, int]) -> np.ndarray:
	wx0, wy0, wx1, wy1 = window_coords
	# np.random.randint is [a, b) is semi-open
	return koop_modes[wx0:wx1, wy0:wy1]


def get_data_win_size(koop_modes: np.ndarray, win_size: tuple[int, int], obst_pos: tuple):
	koop_modes_xy = koop_modes.shape[:2]
	window_coords = gen_window_coord(koop_modes_xy, win_size, obst_pos)

	data_window = get_data_window(koop_modes, window_coords)
	return data_window


def window_coord_centre_point(center: tuple, win_size: tuple[int, int]) -> tuple[int, int, int, int]:
	xc, yc = center
	win_size_x, win_size_y = win_size

	# assuming never can be on top of the obstacle
	# assuming no padding
	wx0 = xc - win_size_x // 2
	wy0 = yc - win_size_y // 2

	wx1 = xc + win_size_x // 2
	wy1 = yc + win_size_y // 2

	return wx0, wy0, wx1, wy1

