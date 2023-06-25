import os

from koolnet.config import get_kool_parent
from koolnet.utils.file_ops import guarantee_existence


def get_data_dir() -> str:
	return guarantee_existence(os.path.join(get_kool_parent(), "data"))


def get_outputs_dir() -> str:
	return guarantee_existence(os.path.join(get_kool_parent(), "outputs"))


def get_plots_dir() -> str:
	return guarantee_existence(os.path.join(get_outputs_dir(), "plots"))
