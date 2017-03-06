import numpy as np

def kernel_lap(x,y,rho,a,b):
	sum = np.sum(abs(x**a-y**a)**b)
	return np.exp(-rho*sum)

