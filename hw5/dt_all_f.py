import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from timeit import default_timer as timer
                     
#--------------------load train and test data------------------------
newdf = pd.read_csv("train.csv",)
newdf_test = pd.read_csv("test.csv")

print()
print(newdf.shape)
print(newdf_test.shape)
print()

print('Label distribution Training set:')
print(newdf['label'].value_counts())

print('\nLabel distribution Testing set:')
print(newdf_test['label'].value_counts())

#-------------------- Split dataframes into X & Y--------------------

# assign X as a dataframe of features and Y as a series of outcome variables

# training set
X_train = newdf.drop('label',1)
Y_train = newdf.label

# test set
X_test = newdf_test.drop('label',1)
Y_test = newdf_test.label

# Save a list of feature names for later use (it is the same for every attack category)
colNames = list(X_train)
colNames_test = list(X_test)


#--------------Use StandardScaler() to scale the dataframes----------

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train) 
X_test = scaler.transform(X_test) 


#---------------------------validation-------------------------------
'''
parameters = {'max_depth':range(3,8),'min_samples_leaf':range(3,8), 'criterion':['gini','entropy']}  #parameters for validation

dtval = GridSearchCV(DecisionTreeClassifier(), parameters, cv=3) 
dtval.fit(X = X_train, y = Y_train)
print()
print(dtval.best_params_)

#----------------------------training-------------------------------
# Build the model for all features:

clf_train = DecisionTreeClassifier(max_depth = dtval.best_params_['max_depth'], min_samples_leaf = dtval.best_params_['min_samples_leaf'], criterion = dtval.best_params_['criterion'])

'''
clf_train = DecisionTreeClassifier(max_depth = 7, min_samples_leaf = 4, criterion = 'entropy' )
clf_train.fit(X = X_train, y = Y_train)


#---------------------Prediction & Evaluation-----------------------

# Apply the classifier we trained to the test data (which it has never seen before)

start_test = timer()    #start timer for testing

Y_test_pred = clf_train.predict(X_test)

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
print("\nAccuracy: %0.5f" % accuracy)
print("Testing time: ",time_test,"sec")

