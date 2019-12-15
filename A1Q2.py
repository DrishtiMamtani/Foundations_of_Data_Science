#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import os 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split

source = "C:/Users/DRISHTI MAMTANI/Desktop/FODS"
dir_list = os.listdir(source)
os.chdir(source)  
df = pd.read_excel('Data.xlsx',header=None)
data = df.values

np.random.seed(1)

X = np.ones((3,data.shape[0]))

# feature scaling
X[:2] = data[:,:2].T # features
mean_X1 = np.sum(X[0])/X.shape[1]
var_X1 = np.std(X[0])
X[0] = (X[0] - mean_X1)/var_X1
mean_X2 = np.sum(X[1])/X.shape[1]
var_X2 = np.std(X[1])
X[1] = (X[1] - mean_X2)/var_X2
Y = data[:,2].T   # labels
mean_Y = np.sum(Y)/Y.shape[0]
var_Y = np.std(Y)
Y = (Y-mean_Y)/var_Y
X = np.array(X)
X = np.reshape(X,(data.shape[0],3))
Y = np.array(Y)
Y = np.reshape(Y,(data.shape[0],1))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

W = np.random.random((3,1)) # weight matrix
#print('W: ',W)
print('X shape:', X_train.shape,'Y shape:',Y_train.shape)
# define hyper parameters
learning_rate = 0.01
number_examples = X_train.shape[0]
epochs = 500
print('training set size:',number_examples)

iteration = []
loss = []

                
w0 = np.random.randn()
print('initial w0 :',w0)
w1 = np.random.randn()
print('initial w1 :',w1)
w2 = np.random.randn()
print('initial w2 :',w2)

W0 = []
W1 = []
Loss = []
iteration = []
for epoch in range(epochs):
    print(epoch)
    dw0 = 0
    dw1 = 0
    dw2 = 0
    loss = 0
    for example in range(number_examples):
        x0 = X_train[example][0]  
        x1 = X_train[example][1]
        x2 = X_train[example][2]
        y  = Y_train[example][0]
        y_tilda = w0*x0 + w1*x1 + w2*x2
        loss += ((y_tilda-y)**2)/(2*number_examples) 
        
        dw0 = (y_tilda - y)*x0
        dw1 = (y_tilda - y)*x1
        dw2 = (y_tilda - y)*x2
        w0 = w0 - learning_rate*dw0/number_examples
        w1 = w1 - learning_rate*dw1/number_examples
        w2 = w2 - learning_rate*dw2/number_examples  
    if epoch%20 ==0:
        W0.append(w0)
        W1.append(w1)
        print('loss: ',loss)
        Loss.append(loss)   
        iteration.append(epoch)   
    for i in range(len(W0)-1):
        if(i>1):
            if (W0[i-1]== W0[i]):
                if(W1[i]==W1[i-1]):
                    break
    #if epoch%50 ==0:
        #print(loss/number_examples)

print('\n\n\n')
print('final w0 :',w0)
print('final w1:',w1)
print('final w2:',w2)
Final_Loss = 0
number_test_examples = X_test.shape[0]
for example in range(number_test_examples):
        x0 = X_test[example][0]  
        x1 = X_test[example][1]
        x2 = X_test[example][2]
        y  = Y_test[example][0]
        y_tilda = w0*x0 + w1*x1 + w2*x2
        Final_Loss += ((y_tilda-y)**2)/(2*number_test_examples) 

print('MSE:',Final_Loss)
plt.figure()
plt.plot(iteration,Loss)
plt.title('Iteration versus Loss')
plt.show()

