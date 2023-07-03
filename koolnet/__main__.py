import koolnet.config
from koolnet import logger

from koolnet.models.random_forest import run_rf_plot_pred
from koolnet.models.xgb_model import run_boost_plot_pred
from koolnet.models.kool_net import run_model


def main():
	args = koolnet.config.parse_args()

	if args.log_level:
		logger.setLevel(args.log_level)

	if args.win_per_mode is None:
		if args.model == "cnn":
			args.win_per_mode = 4000
		else:
			args.win_per_mode = 2000

	if args.model == "rf":
		logger.info(f"Running {args.model} with {args.win_per_mode} windows per mode.")
		run_rf_plot_pred(win_per_mode=args.win_per_mode)

	elif args.model == "xgboost":
		logger.info(f"Running {args.model} with {args.win_per_mode} windows per mode.")
		run_boost_plot_pred(win_per_mode=args.win_per_mode)

	elif args.model == "cnn":
		logger.info(f"Running {args.model} with {args.win_per_mode} windows per mode.")
		run_model(win_per_mode=args.win_per_mode, train=True)


if __name__ == "__main__":
	main()
