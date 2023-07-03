import argparse
import importlib
import inspect
import os

from argparse import Namespace


def get_kool_parent() -> str:
	koolnet_module = importlib.import_module("koolnet")
	koolnet_dir = os.path.dirname(inspect.getabsfile(koolnet_module))
	return os.path.abspath(os.path.join(koolnet_dir, ".."))


def parse_args() -> Namespace:
	parser = argparse.ArgumentParser()

	parser.add_argument(
		"-g",
		action="store_true",
		help="To generate the data only",
	)

	parser.add_argument(
		"-c", "--ckp",
		help="To specify the location of checkpoints"
	)

	parser.add_argument(
		"-m", "--model",
		help="the type of the model to run, can be `rf` / `xgboost` / `cnn`"
	)

	parser.add_argument(
		"-w", "--win-per-mode",
		help="Number of windows per mode to use"
	)

	parser.add_argument(
		"-p", "--plot-graph",
		action="store_true",
		help="Option to plot results.",
	)

	parser.add_argument(
		"--log-level",
		help="Level of the logger, can be DEBUG / INFO / WARNING / ERROR / CRITICAL"
	)

	args = parser.parse_args()
	return args
