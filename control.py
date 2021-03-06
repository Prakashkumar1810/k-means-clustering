import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cluster import Kmeans

dataset = pd.read_csv("data.csv",sep=',',header=None)
X = dataset.iloc[:,:].values

plt.scatter(X[:,0],X[:,1])
plt.show()

model=Kmeans(k=3)
c = model.fit(X)

for eg in X:
	distances = [np.linalg.norm(eg-c[i]) for i in range(3)]
	if distances.index(min(distances))==0:
		plt.scatter(eg[0],eg[1],c='yellow')
	elif distances.index(min(distances))==1:
		plt.scatter(eg[0],eg[1],c='black')
	else:
		plt.scatter(eg[0],eg[1],c='m')

for i in range(3):
	plt.scatter(c[i][0],c[i][1],marker='x',c='r')
plt.show()