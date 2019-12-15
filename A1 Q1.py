import numpy as np
import matplotlib.pyplot as plt
import statistics
from math import sqrt

def Train_Pts(data):     #to extract columns from data
    return data[:,0],data[:,1],data[:,2]

def Loss_Grad(w0,w1,w2,x_train,y_train,z_train): #to calculate loss 

    zcap = w0 + x_train * w1 + y_train * w2
    err = zcap - z_train
    dw0 = np.sum(err)
    dw1 = np.sum(np.multiply(err,x_train))
    dw2 = np.sum(np.multiply(err,y_train))
    errsq = np.square(err)
    E = np.sum(errsq)
    return 0.5*E,dw0,dw1,dw2


def Grad_Des(x_train,y_train,z_train):  #gradient descent algorithm
    #initialising w
    w0,w1,w2 = np.random.normal(),np.random.normal(),np.random.normal()
    loss = []

    eta = 1e-7 #learning rate
    numberOfEpochs = 4000 #stopping criteria
    for x in range(numberOfEpochs): #updating value of w to minimise loss
        Eold,dw0,dw1,dw2 = Loss_Grad(w0,w1,w2,x_train,y_train,z_train)
        loss.append(Eold)
        w0 = w0 - eta * dw0
        w1 = w1 - eta * dw1
        w2 = w2 - eta * dw2

    return w0, w1, w2, loss


def Cal_Error(w0,w1,w2,data): #to calculate error in predicted model
    x,y,z = data[:,0],data[:,1],data[:,2]
    size = len(z)
    zcap = w0 + x * w1 + y * w2
    err = zcap - z
    errsq = np.square(err)
    E = np.sum(errsq) #value of SSE
    z_mean = statistics.mean(z)
    sst = 0
    for pt in z:
        sst = sst + pow(pt-z_mean,2)
    #value of R2 and RMSE
    R2 = 1 - E/sst
    rmse = sqrt(E/size)
        
    return 0.5*E,R2, rmse
    

def main():
    #data preprocesssing
    data = np.loadtxt(open("full_data.txt", "rb"), delimiter=",", skiprows=0)
    data = np.delete(data,0,axis=1)
    data = ( data - np.mean(data, axis= 0) )/np.std(data, axis= 0)
    np.random.shuffle(data)
    
    #spliting train data and test data
    train_size = int(data.shape[0]*80/100)
    test_size = int(data.shape[0]*20/100)
    tr_data = data[0:train_size,:]
    te_data = data[train_size:,:]
    print('traindata',tr_data)
    print('testdata',te_data)
    x_train,y_train,z_train = Train_Pts(tr_data)
    
    #calculating value of w
    w0,w1,w2,loss = Grad_Des(x_train,y_train,z_train)
    print("Weights :",w0,w1,w2)
    
    #calculating error in training data and test data
    E_te, R2_te, rmse_te = Cal_Error(w0,w1,w2,te_data)
    E_tr, R2_tr, rmse_tr = Cal_Error(w0,w1,w2,tr_data)
    print('Test Loss for train data:')
    print("Sum of Squares of Errors =", E_tr)
    print("R2 :", R2_tr)
    print("RMSE :", rmse_tr)
    print('Test Loss for test data :')
    print("Sum of Squares of Errors =", E_te)
    print("R2 :", R2_te)
    print("RMSE :", rmse_te)
    
    #plotting graph of loss vs number of iterations
    plt.plot(loss)
    plt.xlabel('No of Iterations')
    plt.ylabel('Loss Function')
    plt.savefig('GradDes_Plot.png')




if __name__=="__main__":
    main()
