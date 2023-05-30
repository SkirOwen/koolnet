import numpy as np 
import h5py


def load_npz(filename: str) -> dict:
	return np.load(filename)


def load_h5(filename: str) -> tuple:
	with h5py.File(filename, 'r') as file:
		loaded_data = file['data'][:]
		loaded_metadata = dict(file.attrs.items())
	return loaded_data, loaded_metadata
