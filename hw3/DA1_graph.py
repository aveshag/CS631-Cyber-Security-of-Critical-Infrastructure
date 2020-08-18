import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hankel
from scipy.linalg import svd    #library to peform SVD

filename="xmv10_359_data_1.csv"
column_sensor=14
Data=np.loadtxt(filename,delimiter=',') #loading complete data


time_series=Data[0:4800,column_sensor]        #extracting particular sensor data


x1 = np.linspace(0.0, 48.0,num=4800)          #ploting graph for sensor reading
plt.figure(figsize=(10, 5))
plt.plot(x1,time_series) 
plt.xticks(np.arange(0, 50, 5)) 
plt.title('DA1')
plt.xlabel('Time (in hrs)')
plt.ylabel("XMEAS(15)")
plt.savefig('DA1_sensor.png')


train_N=500                 #number of training points

L=train_N//2                #lag parameter

trajectory=hankel(time_series[0:L],time_series[L-1:4800])

train_d=hankel(time_series[0:L],time_series[L-1:train_N])  #trajectory matrix for training


U, ev, VT = svd(train_d)        #singular value decomposition

'''

es = (ev[1:]/(np.sum(ev[1:])))*100
plt.figure(figsize=(20, 20))
plt.plot(es)
plt.title("Statistical Dimension")
plt.xlabel('Number of eigenvalues')
plt.ylabel('Eigenvalue share')
plt.show()

'''


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


test_d=hankel(time_series[4000:L+4000],time_series[L-1+4000:4800])       #trajectory matrix for testing
test_N=4800-4000


dep=np.zeros(4800-L+1-251)
for i in range(251,4800-L+1):                       #checking departure
	x=trajectory[:,i]
	x_des=UrT.dot(x)
	y=c_des-x_des
	D=y.dot(y)
	dep[i-251]=D

x2 = np.linspace(5.0, 48.0,num=4800-L+1-251)
plt.figure(figsize=(10, 5))                     #ploting graph for departure score
plt.plot(x2,dep,label="Departure Score")
plt.axhline(y=thres,color='r', linestyle='--',label="Threshold")
plt.axvline(x=40,color='g', linestyle='--',label="Attack Start")
plt.xticks(np.arange(5, 50, 5))
plt.title('DA1')
plt.xlabel('Time (in hrs)')
plt.ylabel('Departure Score')
plt.legend()
plt.savefig('DA1_attack.png')
