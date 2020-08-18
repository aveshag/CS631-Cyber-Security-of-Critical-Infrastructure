import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn import metrics
from sklearn import tree
from timeit import default_timer as timer
#from sklearn.externals.six import StringIO  
#import pydot


data=pd.read_excel("train.xlsx")        #reading training data
feature_name=list(data.columns.values)
data=data.values
X_train = data[:,1:]                    #separating labels from features
y_train = data[:,0]

data=pd.read_excel("validation.xlsx").values        #reading validation data
X_val = data[:,1:]                    #separating labels from features
y_val = data[:,0]

data=pd.read_excel("test.xlsx").values       #reading data
X_test = data[:,1:]            #separating labels from features
y_test = data[:,0]




#------------------training start----------------------

start_train = timer()    #start timer for training

dttrain = tree.DecisionTreeClassifier(criterion = 'gini', max_depth = 6, min_samples_leaf = 2) 

dttrain.fit(X = X_train, y = y_train)

end_train = timer()           #end timer for training

train_time=end_train-start_train

print("Training time: ",train_time)

#------------------training end------------------------

'''

#------------------validation start--------------------

y_pred = dttrain.predict(X_val)

acc = metrics.accuracy_score(y_val,y_pred)

print("Accuracy on validation: ",acc)

#------------------validation end----------------------

'''

'''
#--------------Generating Decision tree----------------

dot_data = StringIO() 
tree.export_graphviz(dttrain, out_file=dot_data,filled=True,rounded=True,special_characters=True,feature_names=feature_name[1:]) 
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
graph[0].write_png("Decision_tree.png") 
graph[0].write_pdf("Decision_tree.pdf") 

'''

#--------------------testing start---------------------

start_test = timer()    #start timer for testing

y_pred = dttrain.predict(X_test)

conf = metrics.confusion_matrix(y_test, y_pred, labels=[0,1])   #confusion matrix

acc = metrics.accuracy_score(y_test,y_pred)   #accuracy

end_test = timer()      #end timer for testing

time_test = end_test-start_test

print("Testing time: ",time_test)
print("Accuracy on test: ",acc)
print("Confusion Matrix: ",conf)

#--------------------testing end-----------------------

