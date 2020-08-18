import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hankel
from scipy.linalg import svd    #library to peform SVD

filename="xmv9_hundred_data_1.csv"
column_sensor=4
Data=np.loadtxt(filename,delimiter=',') #loading complete data


time_series=Data[0:4800,column_sensor]        #extracting particular sensor data


train_N=500                 #number of training points

L=train_N//2                #lag parameter


train_d=hankel(time_series[0:L],time_series[L-1:train_N])  #trajectory matrix for training


U, ev, VT = svd(train_d)        #singular value decomposition



r=1                        #statistical dimension

Ur = U[:,0:r]

UrT=Ur.T                     #projection matrix



valid_d=hankel(time_series[500:L+500],time_series[L-1+500:4000])        #trajectory matrix for validation

c=np.mean(valid_d,axis=1)    #centroid
c_des=UrT.dot(c)

thres=0                   #threshold

valid_N=4000-500

for i in range(valid_N-L+1):           #computing threshold
	x=valid_d[:,i]
	x_des=UrT.dot(x)
	y=c_des-x_des
	D=y.dot(y)
	if D>thres:
		thres=D
#print(thres)

test_d=hankel(time_series[4000:L+4000],time_series[L-1+4000:4800])       #trajectory matrix for testing
test_N=4800-4000

dep=np.zeros(test_N-L+1)
for i in range(test_N-L+1):                       #checking departure
	x=test_d[:,i]
	x_des=UrT.dot(x)
	y=c_des-x_des
	D=y.dot(y)
	if D>thres:
		print("Attack!")
