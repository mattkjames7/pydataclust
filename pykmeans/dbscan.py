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
			
		#start with a bunch of unclassified class labels
		# The classes will be defined as follows:
		# Noise = -1, unclassified = 0, part of a cluster = 1,2,3...
		self.clusters = np.zeros(self.m,dtype='int32')
		
		#set the starting point for the cluster IDs
		clustID = 1
		
		#loop through each point
		for i in range(0,self.m):

			#if it hasn't been classified yet...
			if self.clusters[i] == 0:
				#check for neighbours
				nbrs = self._RangeQuery(i,epsilon)
				if nbrs.size < minpts:
					#mark as noise if there aren't at least minpts 
					#neighbours within epsilon
					self.clusters[i] = -1
				else:
					#if we get to this bit, then this point must be a core point
					self.clusters[i] = clustID
					j = 0
					while j < nbrs.size:
						if self.clusters[nbrs[j]] == -1:
							#reassign from noise to part of a cluster
							self.clusters[nbrs[j]] = clustID
						elif self.clusters[nbrs[j]] == 0:
							#assign unclassified to part of a cluster
							self.clusters[nbrs[j]] = clustID
							#find neighbours of this point
							jnbrs = self._RangeQuery(nbrs[j],epsilon)
							if jnbrs.size >= minpts:
								use = np.zeros(jnbrs.size,dtype='bool')
								for k in range(0,jnbrs.size):
									use[k] = not jnbrs[k] in nbrs
								use = np.where(jnbrs)[0]
								nbrs = np.append(nbrs,jnbrs[use])
						j+=1
					clustID += 1
					
					
					
	def _RangeQuery(self,i,epsilon):
		'''
		find all of the points within epsilon distance of point i.
		'''
		pt = self.data[i]
		#calculate the distances from all other points
		R = np.linalg.norm(self.data - pt,axis=1)
		#find the indices of the neighbours within range
		use = np.where((R < epsilon) & (self.datainds != i))[0]
		
		return use
		
