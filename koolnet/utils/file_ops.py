import numpy as np 
import h5py



def load_npz(filename: str) -> dict:
	return np.load(filename)