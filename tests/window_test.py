import matplotlib.pyplot as plt
import numpy as np

from koolnet.data.windows import get_data_win_size, get_data_window, window_coord_centre_point


def create_test_array(size: tuple[int, int] = (100, 100)):
	x = np.arange(size[0])
	y = np.arange(size[1])
	arr = np.array([[i + j*100 for i in x] for j in y])
	return arr


def check_window_size(arr, win_size=(10, 10)):
	arr = get_data_win_size(arr, win_size=win_size, obst_pos=(2, 2, 1))
	return arr


def window_from_centre(centre=(17, 8), win_size=(10, 10)) -> None:
	arr = create_test_array()
	plt.imshow(arr)
	plt.fill(
		(1 * np.cos(np.arange(0, 2 * np.pi, 0.01))) + centre[0],
		(1 * np.sin(np.arange(0, 2 * np.pi, 0.01))) + centre[1],
		"r"
	)
	win = window_coord_centre_point(centre, win_size=win_size)
	wx0, wy0, wx1, wy1 = win
	# Window
	plt.fill(
		(np.array([wx0, wx1, wx1, wx0])),
		(np.array([wy0, wy0, wy1, wy1])),
		color=(0.7, 0.2, 0.3, 0.70)
	)
	plt.show()
	print(win)
	data = get_data_window(arr, win)
	print(data.shape)


def main():
	test_arr = create_test_array()
	arr = check_window_size(test_arr)
	plt.imshow(arr)
	plt.show()


if __name__ == "__main__":
	window_from_centre()
	# main()
