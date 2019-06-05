from .RescaleData import RescaleData
from .UnscaleData import UnscaleData
import numpy as np

class kmeans(object):
	def __init__(self,k,data=None,Rescale=True,labels=None):
		'''
		Entry point to the kmeans object - supply at least an integer 
		value for k, optionally supply the data too.
		'''
		
		#store k
		self.k = k
		
		#store data if supplied
		if not data is None:
			self.InsertData(data,Rescale)
		
		
		
		
	def InsertData(data,Rescale=True,labels=None):
		'''
		This will take the data matrix in, rescale optionally, also store
		labels if supplied.
		
		'''
		
		#use the shape to determine number of parameters and samples
		if np.size(data.shape) != 2:
			print("data needs to be a 2-D array, shape (m,n), where m is the number of samples, and n is the number of parameters")
		self.m = data.shape[0]
		self.n = data.shape[1]
		
		#store matrix
		if Rescale:
			self.data,self.scale = RescaleData(data)
		else:
			self.data = data
			self.scale = None
			
		#now we know n, we can calculate the first cluster centroids
		self._CalculateCentroids()
		
	def _CalculateCentroids(self):
		'''
		Randomize positions of k centroids in n dimensional parameter 
		space.
		'''
		
		self.centroids = np.random.random_sample((self.k,self.n))
		
		
	def Train(nSteps=10,Reset=False):
		'''
		Steps the centroids towards the clusters.
		
		'''
		
		#check if we want to start again
		if Reset:
			self._CalculateCentroids()
			
		#loop through taking a step every time
		for i in range(0,nSteps):
			self._TrainStep()
			
	def _TrainStep():
		'''
		Takes a single step
		
		'''
	
	def _FindNearestCentroids():
		'''
		Associates each point with its nearest centroid.
		
		'''
		dist = np.zeros((self.k,self.m))
		
		for i in range(0,self.k):
			dist[i] = np.linalg.norm(self.data - self.centroids[i],axis=1)
			
		self.closest = dist.argmin(axis=0)
