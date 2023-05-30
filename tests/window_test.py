import matplotlib.pyplot as plt
import numpy as np

from koolnet.data.windows import get_data_win_size, get_data_window


def create_test_array(size: tuple[int, int] = (100, 100)):
	x = np.arange(size[0])
	y = np.arange(size[1])
	arr = np.array([[i + j*100 for i in x] for j in y])
	return arr


def check_window_size(arr, win_size=(10, 10)):
	arr = get_data_win_size(arr, win_size=win_size, obst_pos=(2, 2, 1))
	return arr


def main():
	test_arr = create_test_array()
	arr = check_window_size(test_arr)
	plt.imshow(arr)
	plt.show()


if __name__ == "__main__":
	main()
