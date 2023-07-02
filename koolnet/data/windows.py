from __future__ import annotations

import numpy as np


def gen_window_coord(
		xy_size: tuple[int, int],
		win_size: tuple[int, int],
		obst_pos: tuple,
		downstream: bool = False,
		padding: int = 10,
) -> tuple:
	"""
	Gives the coordinates of a rectangular window to slice an array of shape `xy_size` (given as an (x, y) tuple).

	The out is given in the following order: wx0, wy0, wx1, wy1.
	Where (wx0, wy0) correspond to the (x, y) of one corner, and (wx1, wy1) correspond to the (x, y) of the
	diagonally opposite corner.
	The first corner must be the one with the lowest x and y. The second must have the highest.

	Parameters
	----------
	xy_size : tuple of 2 ints
		Tuple of (x, y) corresponding to the shape of the array in which to generate the window.
	win_size : tuple of 2 ints
		Tuple (l, w) to give the length (l) and width (w) of the window to generate.
	obst_pos : tuple of 3 ints
		Tuple (x, y, r) of the position and radius of the obstacle to avoid generating the window on top of it.
	downstream : bool, Optional
		If True, only samples from x bigger than the x position of the obstacle
		plus three times its radius, i.e. (x >= 3 * radius + obst_pos_x). The default is False.
	padding : int, optional
		Determine the amount of padding from the side of the array. The window will need to be `padding` away
		from the side. The default is 10.

	Returns
	-------
	tuple of 4 ints
		Tuples of the four corner of the window.

	Notes
	-----
	This is not seeded, to set a seed it recommended to use np.random.seed(RANDOM_SEED) beforehand.
	"""
	# pick a random points
	# check if in bounds and the cylinder is not in
	# create a padded area of padding == win_size, so guarantee to be in bounds.
	# TODO: set a seed for the random.randint

	x, y = xy_size
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


def get_data_window(arr: np.ndarray, window_coords: tuple[int, int, int, int]) -> np.ndarray:
	"""
	Slice `arr` to extract only the data in `window_coords`.
	The first corner must be the one with the lowest x and y. The second must have the highest.

	Parameters
	----------
	arr : ndarray
		2D ndarray of the data the to slice
	window_coords : tuple of 4 ints
		Tuple of the opposite corner of the window.
		Should be ordered as lowest (x, y) corner followed by the opposite, as:
		wx0, wy0, wx1, wy1

	Returns
	-------
	ndarray
		Sliced of the original array.

	See Also
	--------
	gen_window_coord
	"""
	wx0, wy0, wx1, wy1 = window_coords
	# np.random.randint is [a, b) is semi-open
	return arr[wx0:wx1, wy0:wy1]


def get_data_win_size(arr: np.ndarray, win_size: tuple[int, int], obst_pos: tuple):
	"""
	Helper function to both generate the coordinates of a random window coordinates and extract the data in it.
	The out is given in the following order: wx0, wy0, wx1, wy1.
	Where (wx0, wy0) correspond to the (x, y) of one corner, and (wx1, wy1) correspond to the (x, y) of the
	diagonally opposite corner.
	The first corner must be the one with the lowest x and y. The second must have the highest.


	Parameters
	----------
	arr : ndarray
		The ndarray of the data to use.
	win_size : tuple of 2 ints
		Tuple (l, w) to give the length (l) and width (w) of the window to generate.
	obst_pos : tuple of 3 ints
		Tuple (x, y, r) of the position and radius of the obstacle to avoid generating the window on top of it.
	Returns
	-------
	ndarray
		Sliced of the original array.

	See Also
	--------
	gen_window_coord
	get_data_window
	"""
	koop_modes_xy = arr.shape[:2]
	window_coords = gen_window_coord(koop_modes_xy, win_size, obst_pos)

	data_window = get_data_window(arr, window_coords)
	return data_window


def window_coord_centre_point(centre: tuple[float, float], win_size: tuple[int, int]) -> tuple[int, int, int, int]:
	"""
	Function to generate the window coordinates given the coordinate of the centre point.

	Parameters
	----------
	centre : tuple of 2 floats
		Coordinates (x, y) of the centre.
	win_size : tuple of 2 ints
		Tuple (l, w) to give the length (l) and width (w) of the window to generate.

	Returns
	-------
	tuple of 4 ints
		Tuples of the four corner of the window.

	See Also
	--------
	gen_window_coord
	"""
	xc, yc = centre
	win_size_x, win_size_y = win_size

	# assuming never can be on top of the obstacle
	# assuming no padding
	wx0 = int(xc - win_size_x // 2)
	wy0 = int(yc - win_size_y // 2)

	wx1 = int(xc + win_size_x // 2)
	wy1 = int(yc + win_size_y // 2)

	return wx0, wy0, wx1, wy1

