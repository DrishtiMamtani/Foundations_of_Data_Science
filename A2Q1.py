# -*- coding: utf-8 -*-

import numpy as np
from math import sqrt
import statistics

def TrainPts(data):
    #extracting colums from data
    x, y, z = data[:,0],data[:,1],data[:,2]
        
    return x,y,z
    

def Loss_Grad(w,x_train,y_train,z_train,deg): #calculating loss 

    dw = []
    i = 0
    zcap = 0
    for j in range(deg+1):
        for k in range(j+1):
            zcap = zcap + w[i]*pow(x_train,k)*pow(y_train,j-k)
            i = i+1
    err = zcap - z_train #calculating error 
        
    for j in range(deg+1):
        for k in range(j+1):
            prod = np.multiply(pow(x_train,k),pow(y_train,j-k))
            dw.append(np.sum(np.multiply(err,prod))) #change in w
            
    errsq = np.square(err)
    E_tr = np.sum(errsq) 
        
    return 0.5*E_tr,dw

   
def Grad_Des(x_train,y_train,z_train, deg): #gradient descent algorithm
    w = []
    num_coeff = 0
    #number of terms in w
    num_coeff = int((deg+1)*(deg+2)/2)
    
    #initialising w with random values from standard normal distribution
    for i in range(num_coeff):
        w.append(np.random.normal())
    
    loss = []

    eta = 1e-7 #learning rate
    numberOfEpochs = 4000 #stopping criteria
    for x in range(numberOfEpochs):
        Eold,dw = Loss_Grad(w,x_train,y_train,z_train, deg)
        loss.append(Eold)
        
        for i in range(num_coeff):
            w[i] = w[i] - eta * dw[i] #updating w

    return w, loss


def Cal_Error(w,x,y,z,deg): #calculating error in predicted model
    
    size = len(z)
    i = 0
    z_pred = [0] * size
    
    
    for j in range(deg+1):
        for k in range(j+1):
            z_pred = z_pred + w[i]*pow(x,k)*pow(y,j-k)
            i = i+1
    #calculating SSE
    err = z_pred - z
    errsq = np.square(err)
    E = np.sum(errsq)
    
    #calculating R2
    z_mean = statistics.mean(z)
    st = np.square(z-z_mean)
    sst = np.sum(st)
    sr = np.square(z_pred-z_mean)
    ssr = np.sum(sr)
    R2 = ssr/sst
    
    #calculating RMSE
    rmse = sqrt(E/size)
        
    return 0.5*E,R2, rmse

def main():
    #data preprocessing
    data = np.loadtxt(open("data.txt", "rb"), delimiter=",", skiprows=0)
    data = np.delete(data,0,axis=1)
    data = ( data - np.mean(data, axis= 0) )/np.std(data, axis= 0)
    np.random.shuffle(data)
    
    #splitting data into train data and test data
    tr_size = int(data.shape[0]*80/100)
    te_size = int(data.shape[0]*20/100)
    tr_data = data[0:tr_size,:]
    te_data = data[tr_size:,:]
    
    #Applying gradient descent and calculating error for each model
    x_tr,y_tr,z_tr = TrainPts(tr_data)
    x_te,y_te,z_te = TrainPts(te_data)
    f = open("Results.txt", "w")
    for deg in range(1,7):
        print("\n\nDegree =", deg)
        f.write('\n'+'\n'+'\n'+'Degree ='+repr(deg))
        
        #calculating w and loss
        w,loss = Grad_Des(x_tr,y_tr,z_tr, deg)
        print("Weights = ",w)
        f.write('\n'+'Weights = '+repr(w))
        
        #calculating error in train data
        E_tr, R2_tr, rmse_tr = Cal_Error(w,x_tr,y_tr,z_tr,deg)
        f.write("\n\nTraining Data:")
        print("E_tr =",E_tr)
        f.write('\n'+'Sum of squares of error ='+repr(E_tr))
        print("R2_tr =",R2_tr)
        f.write('\n'+'R^2 = '+repr(R2_tr))
        print("rmse_tr =",rmse_tr)
        f.write('\n'+'RMSE ='+repr(rmse_tr))
        
        #calculating error in test data
        E_te, R2_te, rmse_te = Cal_Error(w,x_te,y_te,z_te,deg)
        f.write("\n\nTest Data:")
        print("Sum of squares of error =",E_te)
        f.write('\n'+'Sum of squares of error ='+repr(E_te))
        print("R^2 = ", R2_te)
        f.write('\n'+'R^2 = '+repr(R2_te))
        print("RMSE =", rmse_te)
        f.write('\n'+'RMSE ='+repr(rmse_te))
    

if __name__=="__main__":
    main()
