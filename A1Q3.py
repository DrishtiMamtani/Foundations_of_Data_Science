#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import os 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from sklearn.model_selection import train_test_split
source = "C:/Users/DRISHTI MAMTANI/Desktop/FODS"
dir_list = os.listdir(source)
os.chdir(source)  
df = pd.read_excel('Data.xlsx',header=None)
data = df.values     #df is a dataframe which is converted to array named data
X = data[:,:2]
Y = data[:,2]
X = X.astype(float) #converting all the values to float
Y = Y.astype(float) 
Y = np.expand_dims(Y,axis=0) #to convert 1D array to 2D array
Y =Y.T #to convert a row 2D row matrix to 2D column matrix
n=len(X) #total n enties are there in the dataset
test_entries=int(n*0.7)
crossvalidate_entries=int(n*0.88)
X_train,X_crossv,X_test = X[:test_entries,:],X[test_entries:crossvalidate_entries,:],X[crossvalidate_entries:,:]
Y_train,Y_crossv,Y_test = Y[:test_entries,:],Y[test_entries:crossvalidate_entries,:],Y[crossvalidate_entries:,:]
mean_train1 = np.mean(X_train,axis=0) #axis=0 to calculate mean of the entire column
std_train1 = np.std(X_train,axis=0)
mean_crossv1 = np.mean(X_crossv,axis=0) 
std_crossv1 = np.std(X_crossv,axis=0)
mean_test1 = np.mean(X_test,axis=0) 
std_test1 = np.std(X_test,axis=0)
X_train=(X_train-mean_train1)/std_train1
X_crossv=(X_crossv-mean_crossv1)/std_crossv1
X_test=(X_test-mean_test1)/std_test1
#X_train,X_test and X_crossv are 2D arrays
#Same is done for Y also
mean_train2 = np.mean(Y_train) 
std_train2 = np.std(Y_train)
mean_crossv2 = np.mean(Y_crossv) 
std_crossv2 = np.std(Y_crossv)
mean_test2 = np.mean(Y_test) 
std_test2 = np.std(Y_test)
Y_train=(Y_train-mean_train2)/std_train2
Y_crossv=(Y_crossv-mean_crossv2)/std_crossv2
Y_test=(Y_test-mean_test2)/std_test2
w0=10
w1=10
w2=10
lambdavalues=np.linspace(0.1,1.4,15)
#L2 Regularization
lossvalues=[]
finalw0=0
finalw1=0
finalw2=0
bestlambda=0
for l in lambdavalues:
    print("hi")
    print("Value of lambda=")
    print(l)
    print("\n")
    derivative1=0
    derivative2=0
    derivative3=0
    for num in range(0,100):
        for i in range(0,len(X_train)):
            derivative1=derivative1+(w0+w1*X_train[i][0]+w2*X_train[i][1]-Y_train[i][0])+l*w0
            derivative2=derivative2+((w0+w1*X_train[i][0]+w2*X_train[i][1]-Y_train[i][0])*X_train[i][0])+l*w1
            derivative3=derivative3+((w0+w1*X_train[i][0]+w2*X_train[i][1]-Y_train[i][0])*X_train[i][1])+l*w2
        w0=w0-(0.0001*derivative1)/len(X_train)
        w1=w1-(0.0001*derivative2)/len(X_train)
        w2=w2-(0.0001*derivative3)/len(X_train)
        print(num)
#         print("ws")
#         print(w0)
#         print(w1)
#         print(w2)
#         print("derivatives")
#         print(derivative1)
#         print(derivative2)
#         print(derivative3)
    loss=0
    print("bye")
    for i in range(0,len(X_crossv)):
        loss=loss+((w0+w1*X_crossv[i][0]*X_crossv[i][1]-Y_crossv[i][0])**2)/(len(X_crossv)*2)
    print(loss)
    if(loss<minloss):
        finalw0=w0
        finalw1=w1
        finalw2=w2
        bestlambda=l
        minloss=loss
    lossvalues.append(loss)   
y=lossvalues
x=lambdavalues
plt.plot(x, y, color='green', linestyle='none', linewidth = 12, 
         marker='o', markerfacecolor='blue', markersize=3) 
plt.xlabel('lambdavalues') 
plt.ylabel('lossvalues') 
plt.savefig('L2Reg1.png')
plt.scatter(x, y, label= "stars", color= "green",  
            marker= "*", s=30) 
plt.savefig('L2Reg2.png')
y_pred=np.zeros(len(X_test))
MSE=0
R2=0
sum
for i in range(0,len(X_test)):
    y_pred[i]=finalw0+finalw1*X_test[i][0]+finalw2*X_test[i][1]
    MSE=MSE+abs(y_pred[i]-Y_test[i][0])**2
RMSE=(np.sqrt(MSE/len(X_test)))    


# In[ ]:


#L1 Regularization
lossvalues=[]
finalw0=0
finalw1=0
finalw2=0
w0=10
w1=10
w2=10
lambdavalues=np.linspace(0.1,1.4,15)
for l in lambdavalues:
    print("hi")
    print("Value of lambda=")
    print(l)
    print("\n")
    derivative1=0
    derivative2=0
    derivative3=0
    for num in range(0,100):
        for i in range(0,len(X_train)):
            derivative1=derivative1+(w0+w1*X_train[i][0]+w2*X_train[i][1]-Y_train[i][0])+(l*np.sign(w0))/2
            derivative2=derivative2+((w0+w1*X_train[i][0]+w2*X_train[i][1]-Y_train[i][0])*X_train[i][0])+(l*np.sign(w1))/2
            derivative3=derivative3+((w0+w1*X_train[i][0]+w2*X_train[i][1]-Y_train[i][0])*X_train[i][1])+(l*np.sign(w2))/2
        w0=w0-(0.0001*derivative1)/len(X_train)
        w1=w1-(0.0001*derivative2)/len(X_train)
        w2=w2-(0.0001*derivative3)/len(X_train)
        print(num)
    loss=0
    print("bye")
    for i in range(0,len(X_crossv)):
        loss=loss+((w0+w1*X_crossv[i][0]*X_crossv[i][1]-Y_crossv[i][0])**2)/(len(X_crossv)*2)
    print(loss)
    if(loss<minloss):
        finalw0=w0
        finalw1=w1
        finalw2=w2
        bestlambda=l
        minloss=loss
    lossvalues.append(loss)   
y_pred=np.zeros(len(X_test))
MSE=0
R2=0
for i in range(0,len(X_test)):
    y_pred[i]=finalw0+finalw1*X_test[i][0]+finalw2*X_test[i][1]
    MSE=MSE+abs(y_pred[i]-Y_test[i][0])**2
RMSE=np.sqrt(MSE)
y=lossvalues
x=lambdavalues
plt.plot(x, y, color='green', linestyle='none', linewidth = 12, 
         marker='o', markerfacecolor='blue', markersize=3) 
plt.xlabel('lambdavalues') 
plt.ylabel('lossvalues') 
plt.savefig('L2Reg1.png')
plt.scatter(x, y, label= "stars", color= "green",  
            marker= "*", s=30) 
plt.savefig('L2Reg2.png')
y_pred=np.zeros(len(X_test))
MSE=0
R2=0
sum
for i in range(0,len(X_test)):
    y_pred[i]=finalw0+finalw1*X_test[i][0]+finalw2*X_test[i][1]
    MSE=MSE+abs(y_pred[i]-Y_test[i][0])**2
RMSE=(np.sqrt(MSE/len(X_test)))

