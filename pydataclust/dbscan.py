import numpy as np
from .RescaleData import RescaleData
from .UnscaleData import UnscaleData

class dbscan(object):
	def __init__(self,data=None,Rescale=True):
		''' 
		Entry point for the dbscan object, data can be entered here.
		'''
		
		if not data is None:
			self.InsertData(data,Rescale)
		else:
			self.data = None
		
	def InsertData(self,data,Rescale=True,labels=None):
		'''
		This will take the data matrix in, rescale optionally, also store
		labels if supplied.
		
		'''
		
		#use the shape to determine number of parameters and samples
		if np.size(data.shape) == 1:
			data = np.array([data]).T
		elif np.size(data.shape) > 2:
			print("data needs to be a 2-D array, shape (m,n), where m is the number of samples, and n is the number of parameters")
			return
		self.m = data.shape[0]
		self.n = data.shape[1]
		
		#store matrix
		if Rescale:
			self.data,self.scales,self.shifts = RescaleData(data)
		else:
			self.data = data
			self.scales = np.ones(self.n)
			self.shifts = np.zeros(self.n)
		self.datainds = np.arange(self.m,dtype='int32')
			
	def ClusterData(self,epsilon=0.1,minpts=4):
		'''
		Cluster the data based on the nearest neighbours.	
		'''
		#check that the data has actually been inserted
		if self.data is None:
			print('Insert data before training, silly!')
			return
			
		#calculate the distance matrix
		self._CalculateDistances(epsilon)
			
		#start with a bunch of unclassified class labels
		# The classes will be defined as follows:
		# Noise = -1, unclassified = 0, part of a cluster = 1,2,3...
		self.clusters = np.zeros(self.m,dtype='int32')
		
		#create empty neighbours array - appending an array takes bloody ages
		#so hopefully this will be quicker
		nbrs = np.zeros(self.m,dtype='int32')
		nn = 0
		
		#set the starting point for the cluster IDs
		clustID = 1
		
		#number of classified points
		nc = 0
		
		#loop through each point
		for i in range(0,self.m):
			print('\rClusters: {:2d}, Classified points: {:6d}'.format(clustID,nc),end='')
			#if it hasn't been classified yet...
			if self.clusters[i] == 0:
				#check for neighbours
				tmp = self._RangeQuery(i,epsilon)
				nn = tmp.size
				nbrs[0:nn] = tmp
				
				if nn < minpts:
					#mark as noise if there aren't at least minpts 
					#neighbours within epsilon
					self.clusters[i] = -1
					nn = 0
					nc += 1
				else:
					#if we get to this bit, then this point must be a core point
					self.clusters[i] = clustID
					nc += 1
					j = 0
					while j < nn:
						if self.clusters[nbrs[j]] == -1:
							#reassign from noise to part of a cluster
							self.clusters[nbrs[j]] = clustID
						elif self.clusters[nbrs[j]] == 0:
							#assign unclassified to part of a cluster
							self.clusters[nbrs[j]] = clustID
							nc += 1
							#find neighbours of this point
							jnbrs = self._RangeQuery(nbrs[j],epsilon)
	
	
							if jnbrs.size >= minpts:
								use = np.zeros(jnbrs.size,dtype='bool')
								for k in range(0,jnbrs.size):
									use[k] = not jnbrs[k] in nbrs[:nn]
								use = np.where(use)[0]
								if use.size > 0:
									nbrs[nn:nn+use.size] = jnbrs[use]
									nn += use.size
						j+=1
						print('\rClusters: {:2d}, Classified points: {:6d}'.format(clustID,nc),end='')
					clustID += 1
		print()		
	
	def _CalculateDistances(self,epsilon):
		'''
		this would hopefully create an array of arrays containing the 
		indices of all the objects within range of each point.
		'''
	
		self._dists = np.zeros((self.m,),dtype='object')
		count = 0
		MB = 1024*1024
		for i in range(0,self.m):
			print('\rCalculating distance matrix {:d} of {:d}, ~ {:6.1f} MB'.format(i+1,self.m,4*count/MB),end='')
			R = np.linalg.norm(self.data - self.data[i],axis=1)
			self._dists[i] = np.where((R < epsilon) & (self.datainds != i))[0]
			count += self._dists[i].size
		print()
		
		
	def _CalculateDistanceMatrix(self,epsilon):
		'''
		This should calculate the distance of all points from each point
		'''
		self._dists = np.zeros((self.m,self.m),dtype='float32')
		self._ineps = np.zeros((self.m,self.m),dtype='bool')
		for i in range(0,self.m):
			print('\rCalculating distance matrix {0} of {1}'.format(i+1,self.m),end='')
			self._dists[i] = np.linalg.norm(self.data - self.data[i],axis=1)
			self._ineps[i] = (self._dists[i] < epsilon) & (self.datainds != i)
		print()
		
						
					
	def _RangeQuery(self,i,epsilon):
		'''
		find all of the points within epsilon distance of point i.
		'''

		#find the indices of the neighbours within range
		use = self._dists[i]
		
		return use
		
