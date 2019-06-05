import numpy as np
import matplotlib.pyplot as plt
from .kmeans import kmeans

def Test(k=3,centx=(0.5,0.1,0.8),centy=(0.25,0.9,1.1),scalex=(0.1,0.3,0.3),scaley=(0.2,0.2,0.4),n=100):
	'''
	Tests the kmeans algorithm
	'''

	x = np.array([])
	y = np.array([])

	plt.figure(figsize=(12,6))
	ax0 = plt.subplot2grid((1,2),(0,0))
	ax1 = plt.subplot2grid((1,2),(0,1))
	for i in range(0,k):
		xt = np.random.randn(n)*scalex[i] + centx[i]
		yt = np.random.randn(n)*scaley[i] + centy[i]
		
		x = np.append(x,xt)
		y = np.append(y,yt)
		
		ax0.scatter(xt,yt)
	
	data = np.array([x,y]).T

	km = kmeans(k,data)
	km.Train()
	
	cent = km.UnscaledCentroids()
	ax0.scatter(cent[:,0],cent[:,1])

	for i in range(0,k):
		use = np.where(km.closest == i)[0]
		ax1.scatter(x[use],y[use])
	ax1.scatter(cent[:,0],cent[:,1])
	
