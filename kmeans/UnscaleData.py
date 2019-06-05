import numpy as np

def UnscaleData(data,scales,shifts):
	'''
	Returns data back to the original scales.
	
	'''
	return data*scales + shifts
