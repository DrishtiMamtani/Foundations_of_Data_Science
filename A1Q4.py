#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
import numpy as np
from sklearn import preprocessing
from numpy.linalg import inv
from sklearn.metrics import r2_score
data=pd.read_excel("Data.xlsx",header=None)
X=data[[0,1]]
y=data[2]
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) 
x_train.reset_index(inplace = True, drop = True) 
x_test.reset_index(inplace = True, drop = True) 
y_train.reset_index(inplace = True, drop = True) 
y_test.reset_index(inplace = True, drop = True) 
n=x_train[0].count()
sum_x1=0
sum_x2=0
sum_y1=0
sum_x1x1=0
sum_x1x2=0
sum_x2x2=0
sum_x1y1=0
sum_x2y1=0
for i in range(0,n):
    sum_x1=sum_x1+x_train[0][i]
    sum_x2=sum_x2+x_train[1][i]
    sum_y1=sum_y1+y_train[i]
    sum_x1x1=sum_x1x1+x_train[0][i]*x_train[0][i]
    sum_x1x2=sum_x1x2+x_train[0][i]*x_train[1][i]
    sum_x2x2=sum_x2x2+x_train[1][i]*x_train[1][i]
    sum_x1y1=sum_x1y1+x_train[0][i]*y_train[i]
    sum_x2y1=sum_x2y1+x_train[1][i]*y_train[i]
A = np.array([[n, sum_x1, sum_x2], [sum_x1, sum_x1x1, sum_x1x2],[sum_x2,sum_x1x2,sum_x2x2]])
B=np.array([sum_y1, sum_x1y1, sum_x2y1])
Inverse=inv(A)
w0=Inverse[0][0]*B[0]+Inverse[0][1]*B[1]+Inverse[0][2]*B[2]
w1=Inverse[1][0]*B[0]+Inverse[1][1]*B[1]+Inverse[1][2]*B[2]
w2=Inverse[2][0]*B[0]+Inverse[2][1]*B[1]+Inverse[2][2]*B[2]
W=np.array([w0,w1,w2])
print(W)
np.around(Inverse @ A).astype(int)
dEbydw0=A[0][0]*w0+A[0][1]*w1+A[0][2]*w2-sum_y1
abs(np.around(dEbydw0))
dEbydw1=A[1][0]*w0+A[1][1]*w1+A[1][2]*w2-sum_x1y1
abs(np.around(dEbydw1))
dEbydw2=A[2][0]*w0+A[2][1]*w1+A[2][2]*w2-sum_x2y1
abs(np.around(dEbydw2))
y_pred=w0+w1*data[1]+w2*data[2]
#print("hi")
MSE=0
y_pred=[]
MSE=0
for i in range(0,len(y_test)):
    y_pred.append(w0+w1*x_test[0][i]+w2*x_test[1][i])
    MSE=MSE+(y_pred[i]-y_test[i])**2
RMSE=np.sqrt(MSE/len(y_test))


