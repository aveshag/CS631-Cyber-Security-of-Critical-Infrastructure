import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


data=pd.read_excel("HW_TESLA.xlt")        #loading data file
feature_name=list(data.columns.values)

data=data.values   
X = data[:,1:]     #separating labels and features          
y = data[:,0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=57)    #spliting training and testing data
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.33,random_state=57)

train=np.c_[y_train.T,X_train]                          #testing data file
df = pd.DataFrame(train,columns=feature_name)
df.to_excel("train.xlsx",index=False)

val=np.c_[y_val.T,X_val]                             #training data file
df = pd.DataFrame(val,columns=feature_name)
df.to_excel("validation.xlsx",index=False)

test=np.c_[y_test.T,X_test]                             #training data file
df = pd.DataFrame(test,columns=feature_name)
df.to_excel("test.xlsx",index=False)
