# -*- coding: utf-8 -*-
"""
Created on Sat Nov 15 17:12:01 2014

@author: Administrator
"""

import sklearn.svm as svm
import numpy as np



def random_line():
    return np.random.rand(4).reshape(2,2)*2-1

def gendata(N=100,line=random_line()):
    def getY(Xs):
        return np.sign(np.cross(line[0,:]-Xs,line[1,:]-line[0,:]))
    Xs = np.random.rand(N*2).reshape(N,2)*2-1
    Ys = getY(Xs)
    return np.hstack((np.ones((N,1)),Xs,Ys.reshape(N,1)))
    
def run_pla(data):
    ## PLA Algorithms
    ## return (#iterations,#learned,#real)
    n,m = np.shape(data)
    Xs = data[:,:-1]
    Ys = data[:,-1]    
    current = np.zeros(m-1)
    niter = 0

    def find_error(current):
        cur_label = np.sign(np.sum(Xs*current, axis=1))
        return [i for i in range(n) if cur_label[i]!=Ys[i]]

    while find_error(current):
        idx = np.random.choice(find_error(current))
        current += Ys[idx]*Xs[idx,:]
        niter+=1
    
    return current

        
def run_svm(data):
    s = svm.SVC(C=10**7,kernel='linear',tol=1e-6)
    Xs = data[:,1:-1]
    Ys = data[:,-1]
    s.fit(Xs,Ys)
    return s
    
    
def run_once(N):
    line = random_line()    
    
    d_train = gendata(N,line)
    while len(set(d_train[:,-1]))==1:
        d_train = gendata(N,line)
        
            
    d_test = gendata(N*100,line)
    w = run_pla(d_train)
    s = run_svm(d_train)
    
    pd_w = np.sign(d_test[:,:-1].dot(w))
    pd_s = s.predict(d_test[:,1:-1])
    y = d_test[:,-1]
    
    return np.sum(pd_s!=y) < np.sum(pd_w!=y)    
    
    
def nsv(N):
    line = random_line()    
    d_train = gendata(N,line)
    while len(set(d_train[:,-1]))==1:
        d_train = gendata(N,line)
    s = run_svm(d_train)
    d = np.abs(s.decision_function(d_train[:,1:-1]))
    return np.sum(d-np.min(d) < 1e-3)
    