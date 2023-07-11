from __future__ import annotations

import os.path

import matplotlib.pyplot as plt
import numpy as np
import cmocean as cm
import seaborn as sns
from matplotlib import pyplot as plt

from koolnet.utils.file_ops import load_h5
from koolnet.data.windows import gen_window_coord
from koolnet.utils.directories import get_plots_dir
from koolnet.data.preprocessing import dist_win_obst


def plot_multiple(
		xi: np.ndarray,
		w_coor: list,
		obst_pos: tuple,
		ys: list,
		title: None | str = None,
		cmap=cm.cm.gray,
		draw_line: bool = False,
		model_name: str = "placeholder",
):
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
		cmap=cmap,
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
		if draw_line:
			axs.fill(
				(np.array([wx0, scale_pred_x])),
				(np.array([wy0, scale_pred_y])),
				color=(0.37, 0.23, 0.17, 0.75)
			)

	axs.set_aspect('equal')
	plt.tight_layout()
	plt.title(title)
	plt.savefig(
		os.path.join(get_plots_dir(), f"{model_name}_{title}_multiplot.png")
		, dpi=400
	)
	plt.show()


def plot_window(data, win_coords, obst_pos, pred, cmap=cm.cm.ice, line: bool = False):
	wx0, wy0, wx1, wy1 = win_coords

	plot_x, plot_y = data.shape[:2]

	# Need to flip axes as Xs are rows and Ys are cols
	x, y = np.meshgrid(np.arange(plot_x), np.arange(plot_y))
	obst_x, obst_y, obst_r = obst_pos

	d = 2 * obst_r
	d = 1

	fig, axs = plt.subplots(figsize=(10, 6))

	contour_1 = np.linspace(np.min(np.real(data)), np.max(np.real(data)), 21)

	c1 = axs.contourf(
		# (x - obst_x) / d,
		# (y - obst_y) / d,
		x,
		y,
		np.real(data.T),
		contour_1,
		cmap=cmap,
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
		color=(0.7, 0.2, 0.3, 0.85)
	)

	scale_pred_x = (wx0 + pred[0])
	scale_pred_y = (wy0 + pred[1])

	# Predicted obstacle
	axs.fill(
		(obst_r * np.cos(np.arange(0, 2 * np.pi, 0.01))) + scale_pred_x,
		(obst_r * np.sin(np.arange(0, 2 * np.pi, 0.01))) + scale_pred_y,
		color=(0.45, 0.23, 0.75, 0.8)
	)
	if line:
		# Line
		axs.fill(
			(np.array([wx0, wx0 + pred[0]])),
			(np.array([wy0, wy0 + pred[1]])),
			color=(0.37, 0.83, 0.17, 0.75)
		)

	print(f"TL Corner: {wx0}, {wy0} ({wy0})")
	print(f"Absolute pos: {scale_pred_x}, {scale_pred_y}")
	print(f"Relative pos: {pred[0]}, {pred[1]}")

	# axs.set_ylim(axs.get_ylim()[::-1])
	axs.set_xticks(np.arange(0, 400, 25))
	axs.set_yticks(np.arange(0, 100, 10))
	axs.set_aspect('equal')
	plt.tight_layout()
	plt.savefig(
		os.path.join(get_plots_dir(), f"one_window_L{int(line)}.png"),
		dpi=400
	)
	plt.show()


def plot_pred_obs_dist(obs, win_coors, pred) -> None:
	xp, yp, rp = obs
	distances = []
	x, y = [], []
	for w, obst_pos in zip(win_coors, pred):
		wx0, wy0, _, _ = w
		obs_x, obs_y = obst_pos
		xo = wx0 + obs_x
		yo = wy0 + obs_y
		x.append(xo)
		y.append(yo)
		distances.append(np.sqrt((xp - xo)**2 + (yp - yo)**2) / rp)

	f, ax = plt.subplots(figsize=(7, 5))
	sns.despine(f)
	sns.histplot(
		distances,
		edgecolor=".3",
		linewidth=.5,
	)
	plt.xlabel("Distance to the obstacle normalised to the radius")
	plt.ylabel("Count")
	plt.savefig("hist_dist_norm_radius.svg", format="svg")
	plt.show()
	f, ax = plt.subplots(figsize=(7, 5))
	sns.despine(f)
	sns.histplot(
		x=(xp - np.array(x)) / (2 * rp),
		y=(yp - np.array(y)) / (2 * rp),
		cbar=True,
		cbar_kws=dict(shrink=.75),
	)
	plt.title("Heatmap of the distance scaled to the diameter of the obstacle")
	plt.savefig("heatmap_dist.svg", format="svg")
	plt.show()


def main():
	data, metadata = load_h5("cylinder_xi_1_50.h5")
	mode = 20
	mode_idx = list(metadata["powers"]).index(mode)
	xi = data[mode_idx]
	# xi = (-1j * xi.reshape(100, 400)).T

	obst_pos = metadata["obst_x"], metadata["obst_y"], metadata["obst_r"]
	# win_coord = [
	# 	obst_pos[0] - obst_pos[2],
	# 	obst_pos[1] - obst_pos[2],
	# 	obst_pos[0] + obst_pos[2],
	# 	obst_pos[1] + obst_pos[2],
	# ]
	# print(f"{win_coord = }")
	xy = data.shape[1::]
	win_coord = [81+(2*11),  10, 390,  90]

	# win_coord = gen_window_coord(xy_size=xy, win_size=(10, 10), obst_pos=obst_pos, downstream=True)
	win_coord = [160,  41, 170,  51]
	dist_x, dist_y = dist_win_obst(obst_pos[:2], win_coord)

	plot_window(xi, win_coord, obst_pos, [dist_x, dist_y], cmap=cm.cm.gray, line=True)


if __name__ == "__main__":
	main()
