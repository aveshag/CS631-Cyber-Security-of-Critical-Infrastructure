import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from timeit import default_timer as timer

data=pd.read_excel("train.xlsx").values        #reading training data
X_train = data[:,1:]                    #separating labels from features
y_train = data[:,0]

data=pd.read_excel("validation.xlsx").values        #reading validation data
X_val = data[:,1:]                      #separating labels from features
y_val = data[:,0]

data=pd.read_excel("test.xlsx").values        #reading testing data
X_test = data[:,1:]                     #separating labels from features
y_test = data[:,0]



#----------------------PCA start-----------------------

start_pca = timer()       #start timer for PCA

#normalizing the data
sc = StandardScaler()    
X_train = sc.fit_transform(X_train)
X_val = sc.transform(X_val)
X_test = sc.transform(X_test)

#PCA on training data for deciding number of components
pca = PCA() 
pca.fit(X_train) 
ratio=pca.explained_variance_ratio_

cum_ratio=0
count=0
thres=0.999   

#selecting number of components

while cum_ratio < thres:
	cum_ratio = cum_ratio + ratio[count]
	count = count + 1

#PCA on training and testing
pca = PCA(n_components = count)
X_train = pca.fit_transform(X_train) 
X_val = pca.fit_transform(X_val) 
X_test = pca.transform(X_test)

end_pca = timer()       #end timer for PCA

PCA_time = end_pca-start_pca

print("PCA time: ",PCA_time)

#---------------------PCA end--------------------------



#-----------------training start-----------------------

start_train = timer()    #start timer for training

svmtrain = svm.SVC(C = 2.5, kernel = 'linear', gamma = 'scale')

svmtrain.fit(X=X_train, y=y_train)

end_train = timer()      #end timer for training

train_time = end_train-start_train

print("Training time: ",train_time)

#-----------------training end-------------------------

'''
#------------------validation start---------------------

y_pred = svmtrain.predict(X_val)

acc = metrics.accuracy_score(y_val,y_pred)   #accuracy

print("Accuracy on validation: ",acc)

#-----------------validation end-----------------------
'''


#------------------testing start-----------------------

start_test = timer()       #start timer for testing

y_pred = svmtrain.predict(X_test)

conf = metrics.confusion_matrix(y_test, y_pred, labels=[0,1])   #confusion matrix

acc = metrics.accuracy_score(y_test,y_pred)   #accuracy

end_test = timer()         #end timer for testing

time_test = end_test-start_test

print("Testing time: ",time_test)
print("Accuracy on test: ",acc)
print("Confusion Matrix: ",conf)

#------------------testing start-----------------------
