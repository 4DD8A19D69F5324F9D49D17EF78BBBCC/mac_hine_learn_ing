# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 16:44:03 2014

@author: Administrator
"""

import numpy as np

data_in = np.loadtxt("in.dta.txt",dtype=float);
data_out = np.loadtxt("out.dta.txt",dtype=float);

def transform(data):
    Xs = data[:,:-1]
    Ys = data[:,-1]    
    n = len(Ys)
    return np.column_stack([np.ones(n),Xs,Xs**2,Xs[:,0]*Xs[:,1],np.abs(Xs[:,0]-Xs[:,1]),np.abs(Xs[:,0]+Xs[:,1]),Ys])


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


data_in_t = transform(data_in)
data_out_t = transform(data_out)

ks = range(-3,3+1)

ws = [lr_with_reg(data_in_t,10**k) for k in ks]

errors_in = [error_rate(data_in_t,w) for w in ws]
errors_out = [error_rate(data_out_t,w) for w in ws]