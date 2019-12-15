#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
data=np.zeros(160)
data
n=160
while True:
    sum=0
    for i in range(0,n):
        x=np.random.randint(0,2)
        data[i]=x
        sum=sum+x
    mean=sum/n
    if mean<0.4 or mean>0.6:
        break
print(mean)
warnings.filterwarnings("ignore")
x=1
def beta(a, b, mew):
    e1 = ss.gamma(a + b)
    e2 = ss.gamma(a)
    e3 = ss.gamma(b)
    e4 = mew ** (a - 1)
    e5 = (1 - mew) ** (b - 1)
    return (e1/(e2*e3)) * e4 * e5

def plot_beta(a, b):
    Ly = []
    Lx = []
    mews = np.mgrid[0:1:100j]
    for mew in mews:
        Lx.append(mew)
        Ly.append(beta(a, b, mew))
    pl.plot(Lx, Ly, label="a=%f, b=%f" %(a,b))
    pl.legend()
a=2
b=3
ims = []
for i in range(0,n):
    if(data[i]==1):
        a=a+1
    else:
        b=b+1
    if(i%20==0):
        plot_beta(a,b)
    if(i==n-1):
        plot_beta(a,b)
    


# In[ ]:


a=2
b=3
for i in range(0,n):
    if(data[i]==1):
        a=a+1
    else:
        b=b+1
plot_beta(a,b)
    

