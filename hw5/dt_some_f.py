import pandas as pd
from sklearn import preprocessing
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from timeit import default_timer as timer
import matplotlib.pyplot as plt

#--------------------load train and test data------------------------
newdf = pd.read_csv("train.csv",)
newdf_test = pd.read_csv("test.csv")

print(newdf.shape)
print(newdf_test.shape)

print('Label distribution Training set:')
print(newdf['label'].value_counts())

print('Label distribution Testing set:')
print(newdf_test['label'].value_counts())

#-------------------- Split dataframes into X & Y--------------------
# assign X as a dataframe of feautures and Y as a series of outcome variables

# training set
X_train = newdf.drop('label',1)
Y_train = newdf.label

# test set
X_test = newdf_test.drop('label',1)
Y_test = newdf_test.label


# Save a list of feature names for later use (it is the same for every attack category)
colNames=list(X_train)
colNames_test=list(X_test)

#--------------Use StandardScaler() to scale the dataframes----------

scaler = preprocessing.StandardScaler().fit(X_train)
X_train=scaler.transform(X_train) 
X_test=scaler.transform(X_test) 


#------------------------Feature Selection---------------------------

# Recursive Feature Elimination, get best features from 41
'''
clf_train = DecisionTreeClassifier(random_state=0)
rfecv_train = RFECV(estimator = clf_train, step = 1, cv = 3, scoring = 'accuracy')
rfecv_train.fit(X_train, Y_train)
X_rfetrain = rfecv_train.transform(X_train)
true=rfecv_train.support_
rfecolindex_train = [i for i, x in enumerate(true) if x]
rfecolname_train = list(colNames[i] for i in rfecolindex_train)
print(rfecolname_train)
print(rfecolindex_train)
'''

# Note:- The number of features to be selected came out to be 14.

rfecolname_train = ['protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'num_compromised', 'count', 'diff_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_rerror_rate']

print(rfecolname_train)

rfecolindex_train= [1, 2, 3, 4, 5, 7, 12, 22, 29, 34, 35, 36, 37, 39]

X_rfetrain = X_train[:,rfecolindex_train]

print(X_rfetrain.shape)

#---------------------------validation-------------------------------
'''
parameters = {'max_depth':range(3,8),'min_samples_leaf':range(3,8), 'criterion':['gini','entropy']}  #parameters for validation

dtval = GridSearchCV(DecisionTreeClassifier(), parameters, cv=3) 
dtval.fit(X = X_rfetrain, y = Y_train)
print(dtval.best_params_)

#----------------------------training-------------------------------
# Build the model for selected features:

clf_rfetrain = DecisionTreeClassifier(max_depth = dtval.best_params_['max_depth'], min_samples_leaf = dtval.best_params_['min_samples_leaf'], criterion = dtval.best_params_['criterion'])
'''

clf_rfetrain = DecisionTreeClassifier(max_depth = 7, min_samples_leaf = 4, criterion = 'entropy')
clf_rfetrain.fit(X = X_rfetrain, y = Y_train)



#---------------------Prediction & Evaluation-----------------------

# Apply the classifier we trained to the test data (which it has never seen before)

# reduce test dataset to 9 features

X_test_red = X_test[:,rfecolindex_train]

print(X_test_red.shape)

start_test = timer()    #start timer for testing

Y_test_pred = clf_rfetrain.predict(X_test_red)

end_test = timer()      #end timer for testing

time_test = end_test-start_test

#0:normal   #1:dos    #2:probe   #3:r2l   #4:u2r

# Create confusion matrix

conf = metrics.confusion_matrix(Y_test, Y_test_pred, labels=[0,1,2,3,4])
accuracy = metrics.accuracy_score(Y_test,Y_test_pred)
print()
print("Confusion Matrix")
print(conf)
print()
print("Accuracy: %0.5f" % accuracy)
print("Testing time: ",time_test)
