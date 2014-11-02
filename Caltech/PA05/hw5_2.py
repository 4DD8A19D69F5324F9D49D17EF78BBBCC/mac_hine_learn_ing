# -*- coding: utf-8 -*-
"""
Created on Sun Nov  2 10:53:28 2014

@author: Administrator
"""

import numpy as np
import math

def random_line():
    return np.random.rand(4).reshape(2,2)*2-1

def gendata(N=100,line=random_line()):
    def getY(Xs):
        return np.sign(np.cross(line[0,:]-Xs,line[1,:]-line[0,:]))
    Xs = np.random.rand(N*2).reshape(N,2)*2-1
    Ys = getY(Xs)
    return np.hstack((np.ones((N,1)),Xs,Ys.reshape(N,1)))


def theta(x):
    return 1./(1.+np.exp(-x))

def error(w,data):
    Xs = data[:,:-1]
    Ys = data[:,-1]
    n,dim = np.shape(Xs)
    return np.sum(np.log(1.+np.exp(-Ys*Xs.dot(w))))/n


def grad(w,data):
    Xs = data[:,:-1]
    Ys = data[:,-1]
    n,dim = np.shape(Xs)
    return -np.mean(Ys.reshape(n,1)*Xs /(1.+np.exp(Ys*Xs.dot(w))).reshape(n,1),axis=0)

def grad_one(w,data):
    Xs = data[:-1]
    Ys = data[-1]
    return -Ys*Xs/(1.+math.exp(Ys*Xs.dot(w)))


def lr_sgd(data,rate=0.01,eps=0.01):
    n,dim = np.shape(data)
    w = np.zeros(dim-1)
    
    def run_a_epoch_sgd():
        cw = w.copy()
        data_p = data[np.random.permutation(n)]
        for pt in data_p:
            cw -= rate*grad_one(cw,pt)
        return cw
    def run_a_epoch_gd():
        cw = w.copy()
        cw-=rate*grad(cw,data)*len(data)
        return cw
    
    iterno=0
    while 1:
        iterno+=1
        cw = run_a_epoch_sgd()
        if np.linalg.norm(cw-w)<eps:
            break
        w=cw
    return cw,iterno
    
def evaluate(N=100,times=100):
    iters_tot = 0
    e_out_tot = 0
    for i in range(times):
        print('runnning ',i,'th experiment')
        line = random_line()
        data = gendata(N,line)
        cw,iterno = lr_sgd(data)
        e_out = error(cw,gendata(N*10,line))
        iters_tot +=iterno
        e_out_tot += e_out
    return iters_tot/times,e_out_tot/times
        
    
    
