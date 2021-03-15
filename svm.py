# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 16:13:04 2021

@author: Mazen
"""



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



df=pd.read_csv("C:\\Users\\Mazen\\Downloads\\Bio-Assignment-2\\heart_failure_clinical_records_dataset.csv")
y=df.iloc[:,-1:]
df = df.drop(['DEATH_EVENT'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(df,y , test_size=0.3, random_state=4)
x_test.to_csv('TestData.csv',index=False)
data2 = pd.read_csv("TestData.csv")
data2=np.matrix(data2)
y_test=np.matrix(y_test)
from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(x_train, y_train.values.ravel())
y_pred = svclassifier.predict(data2)
#y_pred=y_pred.matrix(y_pred)
p=accuracy_score(y_test,y_pred) 
print(p*100)



