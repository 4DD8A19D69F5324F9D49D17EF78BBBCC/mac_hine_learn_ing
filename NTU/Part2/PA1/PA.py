# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 18:01:30 2014

@author: Administrator
"""

import numpy as np
from sklearn import svm

data_train = np.loadtxt("features.train",dtype=float)
data_test = np.loadtxt("features.test",dtype=float)

def extract_one_vs_all(data,x):
    Ys = data[:,0].copy()
    Xs = data[:,1:]
    p = (Ys == x)
    Ys[p]=1
    Ys[~p]=-1
    return np.column_stack([Xs,Ys])


def extract_one_vs_one(data,x,y):
    Ys = data[:,0].copy()
    Xs = data[:,1:]
    
    p1 = (Ys == x)
    p2 = (Ys == y)
    Ys[p1]=1
    Ys[p2]=-1
    return np.column_stack([Xs[p1|p2],Ys[p1|p2]])
    
def train(data,C,Q):
    mysvm = svm.SVC(C=C,kernel='poly',degree=Q)
    mysvm.fit(data[:,:-1],data[:,-1])
    return mysvm

def error(modelsvm,data):
    Yp = modelsvm.predict(data[:,:-1])
    Ys = data[:,-1]
    return np.mean(Yp!=Ys)
    
   

def q15():
    dt = extract_one_vs_all(data_train,0)
    svc = svm.SVC(C=0.01,kernel='linear')
    svc.fit(dt[:,:-1],dt[:,-1])
    print svc.coef_,svc.intercept_,svc.n_support_
    return svc.coef_
    
def q16():    
    nums = [0,2,4,6,8]
    dts = [extract_one_vs_all(data_train,i) for i in nums]
    svcs = [svm.SVC(C=0.01,kernel='poly',degree=2) for i in nums]
    for i in range(len(nums)):
        svcs[i].fit(dts[i][:,:-1],dts[i][:,-1])
    errs = [error(sv,dt) for sv,dt in zip(svcs,dts)]
    print errs
    return svcs
    
def q18():
    Cs = [0.001,0.01,0.1,1,10]
    dt = extract_one_vs_all(data_train,0)
    dtest = extract_one_vs_all(data_test,0)
    svcs = [svm.SVC(C=c,kernel='rbf',gamma=100) for c in Cs]
    for sv,c in zip(svcs,Cs):
        sv.fit(dt[:,:-1],dt[:,-1])
    errs = [error(sv,dtest) for sv in svcs]
    print errs
    return svcs
    
def q19():
    Gs = [1,10,100,1000,10000]
    dt = extract_one_vs_all(data_train,0)
    dtest = extract_one_vs_all(data_test,0)
    svcs = [svm.SVC(C=0.1,kernel='rbf',gamma=g) for g in Gs]
    for sv,c in zip(svcs,Gs):
        sv.fit(dt[:,:-1],dt[:,-1])
    errs = [error(sv,dtest) for sv in svcs]
    print errs
    return svcs
    
    
    
    
def q20():
    Gs = [1,10,100,1000,10000]
    data = extract_one_vs_all(data_train,0)
    c = np.random.choice(len(data),1000)
    v = np.zeros(len(data),dtype=bool)
    v[c]=True
    d1 = data[np.invert(v)]
    d2 = data[v]
    svcs = [svm.SVC(C=0.1,kernel='rbf',gamma=g) for g in Gs]
    for sv in svcs:
        sv.fit(d1[:,:-1],d1[:,-1])
    errs = [(error(sv,d2),i) for i,sv in enumerate(svcs)]
    minerr = min(errs)
    return next(i for i,x in enumerate(errs) if x == minerr)

def q20t():
    res = [0]*10
    for i in range(100):
        r = q20()
        res[r]+=1
        print i,r
    print res

