import numpy as np


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
	wx0, wy0, wx1, wy1 = win_pos
	pred = (wx0 + rel_pred[0]), (wx1 + rel_pred[1]), obst_pos[2]
	return cal_iou(pred, obst_pos)


def avg_rel_iou(rel_preds, obst_pos, win_poss) -> float:
	lst = []
	for pred, w in zip(rel_preds, win_poss):
		lst.append(rel_iou(pred, obst_pos, w))
	return sum(lst) / len(lst)

