import koolnet.config
from koolnet import logger


def main():
	args = koolnet.config.parse_args()

	if args.log_level:
		logger.setLevel(args.log_level)


if __name__ == "__main__":
	main()
