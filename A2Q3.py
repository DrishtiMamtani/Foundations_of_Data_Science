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
w0=w1=w2=w3=w4=w5=w6=w7=w8=w9=w10=w11=w12=w13=w14=w15=w16=w17=w18=w19=w20=w21=w22=w23=w24=w25=w26=w27=10
lambdavalues=np.linspace(0.1,1.4,5)
#L2 Regularization
lossvalues=[]
finalw0=finalw1=finalw2=finalw3=finalw4=finalw5=finalw6=0
finalw7=finalw8=finalw9=finalw10=finalw11=finalw12=finalw13=0
finalw14=finalw15=finalw16=finalw17=finalw18=finalw19=finalw20=0
finalw21=finalw22=finalw23=finalw24=finalw25=finalw26=finalw27=0
bestlambda=0
minloss=1000000000000000
for l in lambdavalues:
    print("hi")
    print("Value of lambda=")
    print(l)
    print("\n")
    derivative0=derivative1=derivative2=derivative3=derivative4=derivative5=derivative6=0
    derivative7=derivative8=derivative9=derivative10=derivative11=derivative12=derivative13=0
    derivative14=derivative15=derivative16=derivative17=derivative18=derivative19=derivative20=0
    derivative21=derivative22=derivative23=derivative24=derivative25=derivative26=derivative27=0
    print("check1")
    for num in range(0,20):
        print("check2")
        for i in range(0,len(X_train)):
            term=w0+w1*X_train[i][0]+w2*X_train[i][1]+w3*(X_train[i][0])**2+w4*(X_train[i][1])**2+w5*(X_train[i][1])*(X_train[i][0])
            +w6*(X_train[i][0])**3+w7*(X_train[i][1])**3+w8*((X_train[i][0])**2)*X_train[i][1]
            +w9*((X_train[i][1])**2)*X_train[i][0]+w10*(X_train[i][0])**4+w11*(X_train[i][1])**4
            +w12*((X_train[i][0])**2)*((X_train[i][1])**2)+w13*((X_train[i][0])**3)*X_train[i][1]
            +w14*((X_train[i][1])**3)*X_train[i][0]+w15*(X_train[i][0])**5+w16*(X_train[i][1])**5
            +w17*((X_train[i][0])**2)*((X_train[i][1])**3)+w18*((X_train[i][1])**2)*((X_train[i][0])**3)
            +w19*((X_train[i][0])**4)*(X_train[i][1])+w20*((X_train[i][1])**4)*(X_train[i][0])
            +w21*(X_train[i][0])**6+w22*(X_train[i][1])**6+w23*((X_train[i][0])**3)*((X_train[i][1])**3)
            +w24*((X_train[i][0])**2)*((X_train[i][1])**4)+w25*((X_train[i][0])**4)*((X_train[i][1])**2)
            +w26*((X_train[i][0])**1)*((X_train[i][1])**5)+w27*((X_train[i][0])**5)*((X_train[i][1])**1)-Y_train[i][0]
            derivative0=derivative0+term+l*w0
            derivative1=derivative1+(term)*X_train[i][0]+l*w1
            derivative2=derivative2+(term)*X_train[i][1]+l*w2
            derivative3=derivative3+(term)*((X_train[i][0])**2)+l*w3
            derivative4=derivative4+(term)*((X_train[i][1])**2)+l*w4
            derivative5=derivative5+(term)*((X_train[i][1])*(X_train[i][0]))+l*w5
            derivative6=derivative6+(term)*((X_train[i][0])**3)+l*w6
            derivative7=derivative7+(term)*((X_train[i][1])**3)+l*w7
            derivative8=derivative8+(term)*(((X_train[i][0])**2)*X_train[i][1])+l*w8
            derivative9=derivative9+(term)*(((X_train[i][1])**2)*X_train[i][0])+l*w9
            derivative10=derivative10+(term)*((X_train[i][0])**4)+l*w10
            derivative11=derivative11+(term)*((X_train[i][1])**4)+l*w11
            derivative12=derivative12+(term)*(((X_train[i][0])**2)*((X_train[i][1])**2))+l*w12
            derivative13=derivative13+(term)*(((X_train[i][0])**3)*X_train[i][1])+l*w13
            derivative14=derivative14+(term)*(((X_train[i][1])**3)*X_train[i][0])+l*w14
            derivative15=derivative15+(term)*((X_train[i][0])**5)+l*w15
            derivative16=derivative16+(term)*((X_train[i][1])**5)+l*w16
            derivative17=derivative17+(term)*(((X_train[i][0])**2)*((X_train[i][1])**3))+l*w17
            derivative18=derivative18+(term)*(((X_train[i][1])**2)*((X_train[i][0])**3))+l*w18
            derivative19=derivative19+(term)*(((X_train[i][0])**4)*((X_train[i][1])**1))+l*w19
            derivative20=derivative20+(term)*(((X_train[i][1])**4)*((X_train[i][0])**1))+l*w20
            derivative21=derivative21+(term)*((X_train[i][0])**6)+l*w21
            derivative22=derivative22+(term)*((X_train[i][1])**6)+l*w22
            derivative23=derivative23+(term)*(((X_train[i][0])**3)*((X_train[i][1])**3))+l*w23
            derivative24=derivative24+(term)*(((X_train[i][0])**2)*((X_train[i][1])**4))+l*w24
            derivative25=derivative25+(term)*(((X_train[i][1])**2)*((X_train[i][0])**4))+l*w25
            derivative26=derivative26+(term)*(((X_train[i][0])**1)*((X_train[i][1])**5))+l*w26
            derivative27=derivative27+(term)*(((X_train[i][0])**5)*((X_train[i][1])**1))+l*w27
        w0=w0-(0.0001*derivative0)/len(X_train)
        w1=w1-(0.0001*derivative1)/len(X_train)
        w2=w2-(0.0001*derivative2)/len(X_train)
        w3=w3-(0.0001*derivative3)/len(X_train)
        w4=w4-(0.0001*derivative4)/len(X_train)
        w5=w5-(0.0001*derivative5)/len(X_train)
        w6=w6-(0.0001*derivative6)/len(X_train)
        w7=w7-(0.0001*derivative7)/len(X_train)
        w8=w8-(0.0001*derivative8)/len(X_train)
        w9=w9-(0.0001*derivative9)/len(X_train)
        w10=w10-(0.0001*derivative10)/len(X_train)
        w11=w11-(0.0001*derivative11)/len(X_train)
        w12=w12-(0.0001*derivative12)/len(X_train)
        w13=w13-(0.0001*derivative13)/len(X_train)
        w14=w14-(0.0001*derivative14)/len(X_train)
        w15=w15-(0.0001*derivative15)/len(X_train)
        w16=w16-(0.0001*derivative16)/len(X_train)
        w17=w17-(0.0001*derivative17)/len(X_train)
        w18=w18-(0.0001*derivative18)/len(X_train)
        w19=w19-(0.0001*derivative19)/len(X_train)
        w20=w20-(0.0001*derivative20)/len(X_train)
        w21=w21-(0.0001*derivative21)/len(X_train)
        w22=w22-(0.0001*derivative22)/len(X_train)
        w23=w23-(0.0001*derivative23)/len(X_train)
        w24=w24-(0.0001*derivative24)/len(X_train)
        w25=w25-(0.0001*derivative25)/len(X_train)
        w26=w26-(0.0001*derivative26)/len(X_train)
        w27=w27-(0.0001*derivative27)/len(X_train)
        print("check3")
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
        loss=loss+(w0+w1*X_crossv[i][0]+w2*X_crossv[i][1]+w3*(X_crossv[i][0])**2+w4*(X_crossv[i][1])**2
                   +w5*(X_crossv[i][1])*(X_crossv[i][0])+w6*(X_crossv[i][0])**3+w7*(X_crossv[i][1])**3
                   +w8*((X_crossv[i][0])**2)*X_crossv[i][1]+w9*((X_crossv[i][1])**2)*X_crossv[i][0]
                   +w10*(X_crossv[i][0])**4+w11*(X_crossv[i][1])**4+w12*((X_crossv[i][0])**2)*((X_crossv[i][1])**2)
                   +w13*((X_crossv[i][0])**3)*X_crossv[i][1]+w14*((X_crossv[i][1])**3)*X_crossv[i][0]
                   +w15*(X_crossv[i][0])**5+w16*(X_crossv[i][1])**5+w17*((X_crossv[i][0])**2)*((X_crossv[i][1])**3)
                   +w18*((X_crossv[i][1])**2)*((X_crossv[i][0])**3)+w19*((X_crossv[i][0])**4)*(X_crossv[i][1])
                   +w20*((X_crossv[i][1])**4)*(X_crossv[i][0])+w21*(X_crossv[i][0])**6+w22*(X_crossv[i][1])**6
                   +w23*((X_crossv[i][0])**3)*((X_crossv[i][1])**3)+w24*((X_crossv[i][0])**2)*((X_crossv[i][1])**4)
                   +w25*((X_crossv[i][0])**4)*((X_crossv[i][1])**2)+w26*((X_crossv[i][0])**1)*((X_crossv[i][1])**5)
                   +w27*((X_crossv[i][0])**5)*((X_crossv[i][1])**1)-Y_crossv[i][0])**2/(len(X_crossv)*2)
    print(loss)
    if(loss<minloss):
        finalw0=w0
        finalw1=w1
        finalw2=w2
        finalw4=w4
        finalw5=w5
        finalw6=w6
        finalw7=w7
        finalw8=w8
        finalw9=w9
        finalw10=w10
        finalw11=w11
        finalw12=w12
        finalw13=w13
        finalw14=w14
        finalw15=w15
        finalw16=w16
        finalw17=w17
        finalw18=w18
        finalw19=w19
        finalw20=w20
        finalw21=w21
        finalw22=w22
        finalw23=w23
        finalw24=w24
        finalw25=w25
        finalw26=w26
        finalw27=w27
        bestlambda=l
        minloss=loss
        print("check5")
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
finalw0=finalw1=finalw2=finalw3=finalw4=finalw5=finalw6=0
finalw7=finalw8=finalw9=finalw10=finalw11=finalw12=finalw13=0
finalw14=finalw15=finalw16=finalw17=finalw18=finalw19=finalw20=0
finalw21=finalw22=finalw23=finalw24=finalw25=finalw26=finalw27=0
bestlambda=0
minloss=1000000000000000
for l in lambdavalues:
    print("hi")
    print("Value of lambda=")
    print(l)
    print("\n")
    derivative0=derivative1=derivative2=derivative3=derivative4=derivative5=derivative6=0
    derivative7=derivative8=derivative9=derivative10=derivative11=derivative12=derivative13=0
    derivative14=derivative15=derivative16=derivative17=derivative18=derivative19=derivative20=0
    derivative21=derivative22=derivative23=derivative24=derivative25=derivative26=derivative27=0
    print("check1")
    for num in range(0,20):
        print("check2")
        for i in range(0,len(X_train)):
            term=w0+w1*X_train[i][0]+w2*X_train[i][1]+w3*(X_train[i][0])**2+w4*(X_train[i][1])**2+w5*(X_train[i][1])*(X_train[i][0])
            +w6*(X_train[i][0])**3+w7*(X_train[i][1])**3+w8*((X_train[i][0])**2)*X_train[i][1]
            +w9*((X_train[i][1])**2)*X_train[i][0]+w10*(X_train[i][0])**4+w11*(X_train[i][1])**4
            +w12*((X_train[i][0])**2)*((X_train[i][1])**2)+w13*((X_train[i][0])**3)*X_train[i][1]
            +w14*((X_train[i][1])**3)*X_train[i][0]+w15*(X_train[i][0])**5+w16*(X_train[i][1])**5
            +w17*((X_train[i][0])**2)*((X_train[i][1])**3)+w18*((X_train[i][1])**2)*((X_train[i][0])**3)
            +w19*((X_train[i][0])**4)*(X_train[i][1])+w20*((X_train[i][1])**4)*(X_train[i][0])
            +w21*(X_train[i][0])**6+w22*(X_train[i][1])**6+w23*((X_train[i][0])**3)*((X_train[i][1])**3)
            +w24*((X_train[i][0])**2)*((X_train[i][1])**4)+w25*((X_train[i][0])**4)*((X_train[i][1])**2)
            +w26*((X_train[i][0])**1)*((X_train[i][1])**5)+w27*((X_train[i][0])**5)*((X_train[i][1])**1)-Y_train[i][0]
            derivative0=derivative0+term+l*np.sign(w0)/2
            derivative1=derivative1+(term)*X_train[i][0]+l*np.sign(w1)/2
            derivative2=derivative2+(term)*X_train[i][1]+l*np.sign(w2)/2
            derivative3=derivative3+(term)*((X_train[i][0])**2)+l*np.sign(w3)/2
            derivative4=derivative4+(term)*((X_train[i][1])**2)+l*np.sign(w4)/2
            derivative5=derivative5+(term)*((X_train[i][1])*(X_train[i][0]))+l*np.sign(w5)/2
            derivative6=derivative6+(term)*((X_train[i][0])**3)+l*np.sign(w6)/2
            derivative7=derivative7+(term)*((X_train[i][1])**3)+l*np.sign(w7)/2
            derivative8=derivative8+(term)*(((X_train[i][0])**2)*X_train[i][1])+l*np.sign(w8)/2
            derivative9=derivative9+(term)*(((X_train[i][1])**2)*X_train[i][0])+l*np.sign(w9)/2
            derivative10=derivative10+(term)*((X_train[i][0])**4)+l*np.sign(w10)/2
            derivative11=derivative11+(term)*((X_train[i][1])**4)+l*np.sign(w11)/2
            derivative12=derivative12+(term)*(((X_train[i][0])**2)*((X_train[i][1])**2))+l*np.sign(w12)/2
            derivative13=derivative13+(term)*(((X_train[i][0])**3)*X_train[i][1])+l*np.sign(w13)/2
            derivative14=derivative14+(term)*(((X_train[i][1])**3)*X_train[i][0])+l*np.sign(w14)/2
            derivative15=derivative15+(term)*((X_train[i][0])**5)+l*np.sign(w15)/2
            derivative16=derivative16+(term)*((X_train[i][1])**5)+l*np.sign(w16)/2
            derivative17=derivative17+(term)*(((X_train[i][0])**2)*((X_train[i][1])**3))+l*np.sign(w17)/2
            derivative18=derivative18+(term)*(((X_train[i][1])**2)*((X_train[i][0])**3))+l*np.sign(w18)/2
            derivative19=derivative19+(term)*(((X_train[i][0])**4)*((X_train[i][1])**1))+l*np.sign(w19)/2
            derivative20=derivative20+(term)*(((X_train[i][1])**4)*((X_train[i][0])**1))+l*np.sign(w20)/2
            derivative21=derivative21+(term)*((X_train[i][0])**6)+l*np.sign(w21)/2
            derivative22=derivative22+(term)*((X_train[i][1])**6)+l*np.sign(w22)/2
            derivative23=derivative23+(term)*(((X_train[i][0])**3)*((X_train[i][1])**3))+l*np.sign(w23)/2
            derivative24=derivative24+(term)*(((X_train[i][0])**2)*((X_train[i][1])**4))+l*np.sign(w24)/2
            derivative25=derivative25+(term)*(((X_train[i][1])**2)*((X_train[i][0])**4))+l*np.sign(w25)/2
            derivative26=derivative26+(term)*(((X_train[i][0])**1)*((X_train[i][1])**5))+l*np.sign(w26)/2
            derivative27=derivative27+(term)*(((X_train[i][0])**5)*((X_train[i][1])**1))+l*np.sign(w27)/2
        w0=w0-(0.0001*derivative0)/len(X_train)
        w1=w1-(0.0001*derivative1)/len(X_train)
        w2=w2-(0.0001*derivative2)/len(X_train)
        w3=w3-(0.0001*derivative3)/len(X_train)
        w4=w4-(0.0001*derivative4)/len(X_train)
        w5=w5-(0.0001*derivative5)/len(X_train)
        w6=w6-(0.0001*derivative6)/len(X_train)
        w7=w7-(0.0001*derivative7)/len(X_train)
        w8=w8-(0.0001*derivative8)/len(X_train)
        w9=w9-(0.0001*derivative9)/len(X_train)
        w10=w10-(0.0001*derivative10)/len(X_train)
        w11=w11-(0.0001*derivative11)/len(X_train)
        w12=w12-(0.0001*derivative12)/len(X_train)
        w13=w13-(0.0001*derivative13)/len(X_train)
        w14=w14-(0.0001*derivative14)/len(X_train)
        w15=w15-(0.0001*derivative15)/len(X_train)
        w16=w16-(0.0001*derivative16)/len(X_train)
        w17=w17-(0.0001*derivative17)/len(X_train)
        w18=w18-(0.0001*derivative18)/len(X_train)
        w19=w19-(0.0001*derivative19)/len(X_train)
        w20=w20-(0.0001*derivative20)/len(X_train)
        w21=w21-(0.0001*derivative21)/len(X_train)
        w22=w22-(0.0001*derivative22)/len(X_train)
        w23=w23-(0.0001*derivative23)/len(X_train)
        w24=w24-(0.0001*derivative24)/len(X_train)
        w25=w25-(0.0001*derivative25)/len(X_train)
        w26=w26-(0.0001*derivative26)/len(X_train)
        w27=w27-(0.0001*derivative27)/len(X_train)
        print("check3")
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
        loss=loss+(w0+w1*X_crossv[i][0]+w2*X_crossv[i][1]+w3*(X_crossv[i][0])**2+w4*(X_crossv[i][1])**2
                   +w5*(X_crossv[i][1])*(X_crossv[i][0])+w6*(X_crossv[i][0])**3+w7*(X_crossv[i][1])**3
                   +w8*((X_crossv[i][0])**2)*X_crossv[i][1]+w9*((X_crossv[i][1])**2)*X_crossv[i][0]
                   +w10*(X_crossv[i][0])**4+w11*(X_crossv[i][1])**4+w12*((X_crossv[i][0])**2)*((X_crossv[i][1])**2)
                   +w13*((X_crossv[i][0])**3)*X_crossv[i][1]+w14*((X_crossv[i][1])**3)*X_crossv[i][0]
                   +w15*(X_crossv[i][0])**5+w16*(X_crossv[i][1])**5+w17*((X_crossv[i][0])**2)*((X_crossv[i][1])**3)
                   +w18*((X_crossv[i][1])**2)*((X_crossv[i][0])**3)+w19*((X_crossv[i][0])**4)*(X_crossv[i][1])
                   +w20*((X_crossv[i][1])**4)*(X_crossv[i][0])+w21*(X_crossv[i][0])**6+w22*(X_crossv[i][1])**6
                   +w23*((X_crossv[i][0])**3)*((X_crossv[i][1])**3)+w24*((X_crossv[i][0])**2)*((X_crossv[i][1])**4)
                   +w25*((X_crossv[i][0])**4)*((X_crossv[i][1])**2)+w26*((X_crossv[i][0])**1)*((X_crossv[i][1])**5)
                   +w27*((X_crossv[i][0])**5)*((X_crossv[i][1])**1)-Y_crossv[i][0])**2/(len(X_crossv)*2)
    print(loss)
    if(loss<minloss):
        finalw0=w0
        finalw1=w1
        finalw2=w2
        finalw4=w4
        finalw5=w5
        finalw6=w6
        finalw7=w7
        finalw8=w8
        finalw9=w9
        finalw10=w10
        finalw11=w11
        finalw12=w12
        finalw13=w13
        finalw14=w14
        finalw15=w15
        finalw16=w16
        finalw17=w17
        finalw18=w18
        finalw19=w19
        finalw20=w20
        finalw21=w21
        finalw22=w22
        finalw23=w23
        finalw24=w24
        finalw25=w25
        finalw26=w26
        finalw27=w27
        bestlambda=l
        print("check5")
    lossvalues.append(loss)   
y=lossvalues
x=lambdavalues
plt.plot(x, y, color='green', linestyle='none', linewidth = 12, 
         marker='o', markerfacecolor='blue', markersize=3) 
plt.xlabel('lambdavalues') 
plt.ylabel('lossvalues') 
plt.savefig('L2Reg1.png')
y_pred=np.zeros(len(X_test))
MSE=0
R2=0
sum
for i in range(0,len(X_test)):
    y_pred[i]=finalw0+finalw1*X_test[i][0]+finalw2*X_test[i][1]
    MSE=MSE+abs(y_pred[i]-Y_test[i][0])**2
    
RMSE=(np.sqrt(MSE/len(X_test)))

