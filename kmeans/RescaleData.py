import numpy as np

def RescaleData(data):
	'''
	Very simple rescaling of data: x' = (x - x_min)/(x_max - x_min)
	'''
	
	mn = np.nanmin(data,axis=0)
	mx = np.nanmax(data,axis=0)
	
	scales = (mx - mn)
	shifts = mn
	
	return (data - shifts)/scales,scales,shifts
