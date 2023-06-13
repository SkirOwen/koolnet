import logging

from .logger import make_logger

__all__ = ["logger", "RANDOM_SEED"]

logger: logging.Logger

logger = make_logger(
	level="INFO"
)

RANDOM_SEED = 17
