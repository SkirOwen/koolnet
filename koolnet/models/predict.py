from __future__ import annotations


from koolnet.utils.metrics import rel_iou


def chain_predict_one(window_coor, model, obstacle, data, allmodes):
	"""if output 0 converges right away
	-1 out-of-bound right away
	negative out-of-bound in n step
	postive converges in n step
	"""
	koopmodes_xy = data.shape[1::]

	for mode in allmodes:
		pass

	div_count = 0
	score = 0
	while score <= 0:
		y = model.predict(X)
		score = rel_iou(y, obstacle, window_coor)
		if not (0 <= y[0] <= koopmodes_xy[0]) or not (0 <= y[1] <= koopmodes_xy[1]):
			div_count = -div_count
			break
		div_count += 1
		# new window
	return div_count - 1


def main():
	pass


if __name__ == "__main__":
	main()
