# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 16:44:03 2014

@author: Administrator
"""

import numpy as np
import sklearn as lr

data_in = np.loadtxt("in.dta.txt",dtype=float);
data_out = np.loadtxt("out.dta.txt",dtype=float);

def transform(data,k):
    Xs = data[:,:-1]
    Ys = data[:,-1]    
    n = len(Ys)
    transformed_Xs = np.column_stack([np.ones(n),Xs,Xs**2,Xs[:,0]*Xs[:,1],np.abs(Xs[:,0]-Xs[:,1]),np.abs(Xs[:,0]+Xs[:,1])])
    return np.column_stack([transformed_Xs[:,:k],Ys])


def lr_with_reg(data, lambda_=0):
    Xs = data[:,:-1]
    Ys = data[:,-1]
    n,m = np.shape(Xs); 
    return np.linalg.inv(Xs.T.dot(Xs)+ lambda_* np.eye(m)).dot(Xs.T).dot(Ys)
    
    


def error_rate(data,w):
    Xs = data[:,:-1]    
    Ys = data[:,-1]
    Yp = np.sign(Xs.dot(w))
    return np.mean(Yp!=Ys)


data_train = data_in[:25,:]
data_validate = data_in[25:,:]

ks = np.arange(3,9)

data_in_t = [transform(data_train,k) for k in ks]
data_val_t = [transform(data_validate,k) for k in ks] 
data_out_t = [transform(data_out,k) for k in ks]

ws = [lr_with_reg(data,0) for data in data_in_t]
error_vals = [ error_rate(data,w)  for data,w in zip(data_val_t,ws)]
error_tests = [ error_rate(data,w)  for data,w in zip(data_out_t,ws)]

print(list(zip(ks,error_vals,error_tests)))




e = np.random.rand(1000000)
e2 = np.random.rand(1000000)
print(np.mean(np.minimum(e,e2)))



def calc(p):
    return (p**2+1)/(p+1)**2/(p-1)**2  - 1/16
    

