from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def cal_iou(pred, obst_pos) -> float:
	# assuming same radius
	xp, yp, rp = pred
	xo, yo, ro = obst_pos
	area_p = np.pi * rp ** 2
	area_o = np.pi * ro ** 2

	d = np.sqrt((xp - xo)**2 + (yp - yo)**2)

	if d >= rp + ro:
		iou = 0
	elif d <= np.abs(rp - ro):
		iou = 1
	else:
		ai = (
			rp**2 * np.arccos((d**2 + rp**2 - ro**2) / (2 * d * rp)) +
			ro**2 * np.arccos((d**2 + ro**2 - rp**2) / (2 * d * ro)) -
			0.5 * np.sqrt((-d + rp + ro) * (d + rp - ro) * (d - rp + ro) * (d + rp + ro))
		)
		au = area_p + area_o - ai

		iou = ai / au

	return iou


def rel_iou(rel_pred, obst_pos, win_pos) -> float:
	# TODO: assuming same size for pred as obst
	wx0, wy0, _, _ = win_pos
	pred = (wx0 + rel_pred[0]), (wy0 + rel_pred[1]), obst_pos[2]
	return cal_iou(pred, obst_pos)


def avg_rel_iou(rel_preds, obst_pos, win_poss, filename: None | str = None) -> float:
	lst = []
	for pred, w in zip(rel_preds, win_poss):
		lst.append(rel_iou(pred, obst_pos, w))
	print(f"Number of non-zero IoU: {len([i for i in lst if i != 0])}")
	print(f"Number of zero IoU: {len([i for i in lst if i == 0])}")
	f, ax = plt.subplots(figsize=(7, 5))
	sns.despine(f)
	sns.histplot(
		lst,
		palette="light:m_r",
		edgecolor=".3",
		linewidth=.5,
	)
	plt.xlabel("IoU")
	# sns.set_theme()
	# plt.hist(lst)
	plt.title("Histogram of IoUs")
	if filename is not None:
		plt.savefig(f"{filename}.svg", format="svg")
	plt.show()
	return sum(lst) / len(lst)


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

