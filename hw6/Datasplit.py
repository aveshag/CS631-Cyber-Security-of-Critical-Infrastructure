import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


data=pd.read_excel("HW_TESLA.xlt")        #loading data file
feature_name=list(data.columns.values)

data=data.values   
X_0 = data[0:1636,1:]     #separating labels and features          
y_0 = data[0:1636,0]

X_1 = data[1637:,1:]     #separating labels and features          
y_1 = data[1637:,0]

X_train, X_test, y_train, y_test = train_test_split(X_0, y_0, test_size=0.25,random_state=57)    #spliting training and testing data

X_test=np.r_[X_1,X_test]
y_test=np.r_[y_1,y_test]

train=np.c_[y_train.T,X_train]                          #testing data file
df = pd.DataFrame(train,columns=feature_name)
df.to_excel("train.xlsx",index=False)

test=np.c_[y_test.T,X_test]                             #training data file
df = pd.DataFrame(test,columns=feature_name)
df.to_excel("test.xlsx",index=False)
