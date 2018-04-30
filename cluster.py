import numpy as np

class Kmeans:
	def __init__(self,k=2,max_iter=300):
		self.k = k
		self.max_iter = max_iter
	
	def fit(self,data):
		self.data=data
		self.centroids={}
		for i in range(self.k):
			self.centroids[i]=data[i]
		
		for epoch in range(self.max_iter):
			clust={}
			for i in range(self.k):
				clust[i]=[]
			
			for x in data:
				distances = [np.linalg.norm(x-self.centroids[centroid]) for centroid in range(self.k)]
				clust[distances.index(min(distances))].append(x)
			
			for i in range(self.k):
				self.centroids[i] = np.average(clust[i],axis=0)
			
		return self.centroids