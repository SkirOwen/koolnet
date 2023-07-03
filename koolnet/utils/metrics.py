from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from typing import Sequence


def iou(pred: Sequence[float, float, float], obst_pos: Sequence[float, float, float]) -> float:
	# assuming same radius
	xp, yp, rp = pred
	xo, yo, ro = obst_pos
	area_p = np.pi * rp ** 2
	area_o = np.pi * ro ** 2

	d = np.sqrt((xp - xo)**2 + (yp - yo)**2)

	if d >= rp + ro:
		iou_val = 0
	elif d <= np.abs(rp - ro):
		iou_val = 1
	else:
		ai = (
			rp**2 * np.arccos((d**2 + rp**2 - ro**2) / (2 * d * rp)) +
			ro**2 * np.arccos((d**2 + ro**2 - rp**2) / (2 * d * ro)) -
			0.5 * np.sqrt((-d + rp + ro) * (d + rp - ro) * (d - rp + ro) * (d + rp + ro))
		)
		au = area_p + area_o - ai

		iou_val = ai / au

	return iou_val


def rel_iou(rel_pred: Sequence, obst_pos: Sequence, win_pos: Sequence) -> float:
	if len(rel_pred) == 2:
		radius_pred = obst_pos[2]
	else:
		radius_pred = rel_pred[1]

	wx0, wy0, _, _ = win_pos
	pred = (wx0 + rel_pred[0]), (wy0 + rel_pred[1]), radius_pred
	return iou(pred, obst_pos)


def avg_rel_iou(rel_preds: Sequence, obst_pos: Sequence, win_poss: Sequence, filename: None | str = None) -> list:
	lst = []
	for pred, w in zip(rel_preds, win_poss):
		lst.append(rel_iou(pred, obst_pos, w))
	f, ax = plt.subplots(figsize=(7, 5))
	sns.despine(f)
	sns.histplot(
		lst,
		edgecolor=".3",
		linewidth=.5,
	)
	plt.xlabel("IoU")
	plt.title("Histogram of IoUs")
	if filename is not None:
		plt.savefig(f"{filename}.svg", format="svg")
	plt.show()
	# print(f"Number of non-zero IoU: {len([i for i in lst if i != 0])}")
	# print(f"Number of zero IoU: {len([i for i in lst if i == 0])}")
	return lst


def shap_expl(model, X_train):
	import shap
	explainer = shap.Explainer(model)
	shap_values = explainer(X_train)

	# visualize the first prediction's explanation
	shap.plots.waterfall(shap_values[0])
	# visualize the first prediction's explanation with a force plot
	shap.plots.force(shap_values[0])
	# visualize all the training set predictions
	shap.plots.force(shap_values)
	# visualize all the training set predictions
	shap.plots.force(shap_values)
	# summarize the effects of all the features
	shap.plots.beeswarm(shap_values)
	shap.plots.bar(shap_values)

