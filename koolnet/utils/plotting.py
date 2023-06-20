import matplotlib.pyplot as plt
import numpy as np

from koolnet.utils.file_ops import load_h5
from koolnet.data.windows import gen_window_coord


def plot_multiple(xi: np.ndarray, w_coor: list, obst_pos: tuple, ys: list):
	plot_x, plot_y = xi.shape[:2]

	# Need to flip axes as Xs are rows and Ys are cols
	x, y = np.meshgrid(np.arange(plot_x), np.arange(plot_y))
	obst_x, obst_y, obst_r = obst_pos

	d = 2 * obst_r
	d = 1

	fig, axs = plt.subplots(figsize=(10, 6))

	contour_1 = np.linspace(np.min(np.real(xi)), np.max(np.real(xi)), 21)

	c1 = axs.contourf(
		# (x - obst_x) / d,
		# (y - obst_y) / d,
		x,
		y,
		np.real(xi.T),
		contour_1,
		# cmap=cm.cm.curl,
		# vmin=np.min(contourp_1),
		# vmax=np.max(contourp_1)
	)
	# cbar1 = fig.colorbar(c1, ax=axs)
	# cbar1.formatter.set_powerlimits((-2, 2))  # Display colorbar tick labels in scientific notation
	# cbar1.update_ticks()

	# Obstacle
	axs.fill(
		obst_r * np.cos(np.arange(0, 2 * np.pi, 0.01)) + obst_x,
		obst_r * np.sin(np.arange(0, 2 * np.pi, 0.01)) + obst_y,
		"r"
	)
	for w, pred in zip(w_coor, ys):
		wx0, wy0, wx1, wy1 = w
		# Window
		axs.fill(
			(np.array([wx0, wx1, wx1, wx0])),
			(np.array([wy0, wy0, wy1, wy1])),
			color=(0.7, 0.2, 0.3, 0.15)
		)
		scale_pred_x = (wx0 + pred[0])
		scale_pred_y = (wy0 + pred[1])

		# Predicted obstacle
		axs.fill(
			(obst_r * np.cos(np.arange(0, 2 * np.pi, 0.01))) + scale_pred_x,
			(obst_r * np.sin(np.arange(0, 2 * np.pi, 0.01))) + scale_pred_y,
			color=(0.45, 0.23, 0.75, 0.2)
		)

	axs.set_aspect('equal')
	plt.tight_layout()
	plt.show()


def plot_window(data, win_coords, obst_pos, pred):
	wx0, wy0, wx1, wy1 = win_coords

	plot_x, plot_y = data.shape[:2]

	# Need to flip axes as Xs are rows and Ys are cols
	x, y = np.meshgrid(np.arange(plot_x), np.arange(plot_y))
	obst_x, obst_y, obst_r = obst_pos

	d = 2 * obst_r
	d = 1

	fig, axs = plt.subplots(figsize=(10, 10))

	contour_1 = np.linspace(np.min(np.real(data)), np.max(np.real(data)), 21)

	c1 = axs.contourf(
		# (x - obst_x) / d,
		# (y - obst_y) / d,
		x,
		y,
		np.real(data.T),
		contour_1,
		# cmap=cm.cm.curl,
		# vmin=np.min(contourp_1),
		# vmax=np.max(contourp_1)
	)
	# cbar1 = fig.colorbar(c1, ax=axs)
	# cbar1.formatter.set_powerlimits((-2, 2))  # Display colorbar tick labels in scientific notation
	# cbar1.update_ticks()

	# Obstacle
	axs.fill(
		obst_r * np.cos(np.arange(0, 2 * np.pi, 0.01)) + obst_x,
		obst_r * np.sin(np.arange(0, 2 * np.pi, 0.01)) + obst_y,
		"r"
	)
	# Window
	axs.fill(
		(np.array([wx0, wx1, wx1, wx0])),
		(np.array([wy0, wy0, wy1, wy1])),
		color=(0.7, 0.2, 0.3, 0.65)
	)

	scale_pred_x = (wx0 + pred[0])
	scale_pred_y = (wy0 + pred[1])

	# Predicted obstacle
	axs.fill(
		(obst_r * np.cos(np.arange(0, 2 * np.pi, 0.01))) + scale_pred_x,
		(obst_r * np.sin(np.arange(0, 2 * np.pi, 0.01))) + scale_pred_y,
		color=(0.45, 0.23, 0.75, 0.8)
	)

	# Line
	axs.fill(
		(np.array([wx0, wx0 + pred[0]])),
		(np.array([wy0, wy0 + pred[1]])),
		color=(0.37, 0.23, 0.17, 0.75)
	)

	print(f"TL Corner: {wx0}, {wy0} ({wy0})")
	print(f"Absolute pos: {scale_pred_x}, {scale_pred_y}")
	print(f"Relative pos: {pred[0]}, {pred[1]}")

	# axs.set_ylim(axs.get_ylim()[::-1])
	axs.set_xticks(np.arange(0, 400, 25))
	axs.set_yticks(np.arange(0, 100, 10))
	axs.set_aspect('equal')
	plt.tight_layout()
	# plt.savefig("./output/plots/fig.svg")
	plt.show()


def main():
	data, metadata = load_h5("xi_v2.h5")
	mode = 18
	mode_idx = list(metadata["powers"]).index(mode)
	xi = data[mode_idx]
	# xi = (-1j * xi.reshape(100, 400)).T
	xy = xi.shape[:2]

	obst_pos = metadata["obst_x"], metadata["obst_y"], metadata["obst_r"]
	# win_coord = [
	# 	obst_pos[0] - obst_pos[2],
	# 	obst_pos[1] - obst_pos[2],
	# 	obst_pos[0] + obst_pos[2],
	# 	obst_pos[1] + obst_pos[2],
	# ]
	# print(f"{win_coord = }")

	# win_coord = gen_window_coord(koop_modes_xy=xy, win_size=(10, 10), obst_pos=obst_pos)
	win_coord = [160,  41, 170,  51]
	plot_window(xi, win_coord, obst_pos, [-79, 12])


if __name__ == "__main__":
	main()
