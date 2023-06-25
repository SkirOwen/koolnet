import os

import numpy as np
import h5py


def guarantee_existence(path: str) -> str:
	"""Function to guarantee the existence of a path, and returns its absolute path.

	Parameters
	----------
	path : str
		Path (in str) to guarantee the existence.

	Returns
	-------
	str
		The absolute path.
	"""
	if not os.path.exists(path):
		os.makedirs(path)
	return os.path.abspath(path)


def load_npz(filename: str) -> dict:
	return np.load(filename)


def load_h5(filename: str) -> tuple:
	with h5py.File(filename, 'r') as file:
		loaded_data = file['data'][:]
		loaded_metadata = dict(file.attrs.items())
	return loaded_data, loaded_metadata
