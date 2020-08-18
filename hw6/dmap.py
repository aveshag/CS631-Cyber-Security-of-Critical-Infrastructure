import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

#-------------------------reading data--------------------------#
data=pd.read_excel("HW_TESLA.xlt")
feature_name=list(data.columns.values)
data=data.values
X = data[:,1:]                    #separating labels from features
y = data[:,0]

print("Dimension of data: ",X.shape)


#------------------------kernel matrix--------------------------#

kernel = metrics.pairwise.rbf_kernel(X, Y = None, gamma = (10/132))


#------------------------Diagonal matrix------------------------#
D = kernel.sum(axis=1)

D_half_n = D**(-1/2)
D_half_p = D**(1/2)
D_half_n_mat = np.diag(D_half_n)


#--------------------Diffusion matrix (P`)----------------------#

temp = np.dot(D_half_n_mat, kernel) 
P_d = np.dot(temp, D_half_n_mat) 

print("Dimension of P`: ",P_d.shape)

#-------------------Eigen values-vectors of P`------------------#
print()
print("calculating eigen values and vectors ...")



w , v = np.linalg.eigh(P_d)
v_new = v * D_half_n[:, None]
#print(v_new)

#-------------------Dataset in Diffusion space------------------#

X_ = v_new * (w.T)

#--------Kmeans on complete Dataset in Diffusion space----------#
print()
print("performing clustering on data with all dimensions ...")
kmeans = KMeans(n_clusters=2, max_iter = 200) 
kmeans.fit(X_)

# KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
#     n_clusters=2, n_init=10, n_jobs=1, precompute_distances='auto',
#     random_state=None, tol=0.0001, verbose=0)


correct = 0
for i in range(len(X_)):
	predict_me = np.array(X_[i].astype(float))
	predict_me = predict_me.reshape(-1, len(predict_me))
	prediction = kmeans.predict(predict_me)
	prediction[0] = 1 - prediction[0]
	if prediction[0] == y[i]:
		correct += 1

print("Accuracy(all dimensions): ",correct/len(X_))

#------------Kmeans on 3D Dataset in Diffusion space------------#
print()
print("performing clustering on data with 3 dimensions ...")
X_3 = X_[:,-3:]

kmeans.fit(X_3)

correct = 0
for i in range(len(X_3)):
	predict_me = np.array(X_3[i].astype(float))
	predict_me = predict_me.reshape(-1, len(predict_me))
	prediction = kmeans.predict(predict_me)
	prediction[0] = 1 - prediction[0]
	if prediction[0] == y[i]:
		correct += 1

print("Accuracy(three dimensions): ",correct/len(X_))

'''
#----------------------------Plotting---------------------------#
fig1=plt.figure()
ax1=fig1.add_subplot(111,projection='3d')
ax1.scatter(X_3[:,0],X_3[:,1],X_3[:,2], marker='s',c=y ,cmap='RdBu')
plt.savefig('original_clusters.png')
fig2=plt.figure()
ax2=fig2.add_subplot(111,projection='3d')
ax2.scatter(X_3[:,0],X_3[:,1],X_3[:,2], marker='s',c=kmeans.labels_, cmap='RdBu')
plt.savefig('after_clustering.png')
'''
