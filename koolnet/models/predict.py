from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
import lightning as pl

from koolnet import logger
from koolnet.utils.metrics import rel_iou
from koolnet.data.windows import window_coord_centre_point
from koolnet.data.preprocessing import get_data_modes, dist_win_obst


def chain_predict_one(window_coor, model, obst_pos, data, allmodes, for_rf):
	"""if output 0 converges right away
	-1 out-of-bound right away
	negative out-of-bound in n step
	postive converges in n step
	"""
	koopmodes_xy = data.shape[1::]
	obst_xy = obst_pos[:2]

	dist_x, dist_y = dist_win_obst(obst_xy, window_coor)
	y = [dist_x, dist_y]

	div_count = 0
	score = 0
	k = 0

	while score <= 0:
		x_data = []
		x_mode_r, x_mode_abs = get_data_modes(allmodes, data, window_coor, for_rf)
		x_data.append((*x_mode_r, *x_mode_abs))
		x_data = np.array(x_data)

		if isinstance(model, pl.LightningModule):
			logger.debug("Chain predict on KoolNet")
			x = torch.Tensor(x_data)
			y = model(x).detach()[0].tolist()
		else:
			y = model.predict(x_data)[0]

		score = rel_iou(y, obst_pos, window_coor)

		if score > 0:
			div_count += 1
			break

		true_y = (window_coor[0] + y[0]), (window_coor[1] + y[1])
		if not (5 <= true_y[0] <= koopmodes_xy[0] - 5) or not (5 <= true_y[1] <= koopmodes_xy[1] - 5):
			# print("Ha")
			div_count = -div_count
			break
		k += 1
		if k > 100:
			logger.debug(f"Chain predict did not converge nor diverged in {k} iterations, stopping the chain")
			break

		div_count += 1
		window_coor = window_coord_centre_point(
			true_y,
			win_size=(10, 10),
		)

	return div_count - 1


def chain_multiple(windows, model, obst_pos, data, allmodes, for_rf):
	rslt = []
	for w in windows:
		div_count = chain_predict_one(w, model, obst_pos, data, allmodes, for_rf)
		rslt.append(div_count)

	labels, counts = np.unique(rslt, return_counts=True)
	print(labels)
	print(counts)
	plt.bar(labels, counts, align="center")
	plt.ylabel("Counts")
	plt.xlabel("Chain prediction convergence")
	plt.savefig("Chain_pred.svg", format="svg")
	plt.show()
	return rslt


def main():
	pass


if __name__ == "__main__":
	main()
