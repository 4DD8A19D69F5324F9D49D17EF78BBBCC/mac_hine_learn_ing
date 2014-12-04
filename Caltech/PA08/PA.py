# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 18:01:30 2014

@author: Administrator
"""

import numpy as np
from sklearn import svm
import itertools

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





def do_experiment1(digits):
    datas = [extract_one_vs_all(data_train,d) for d in digits]
    models = [train(data,C=0.01,Q=2) for data in datas]
    e_ins = [error(model,data) for model,data in zip(models,datas)]
    nsvs = [len(model.support_) for model in models]
    print(list(zip(digits,e_ins,nsvs)))

def do_experiment2(Cs,Qs):
    dtrain = extract_one_vs_one(data_train,1,5)
    dtest = extract_one_vs_one(data_test,1,5)
    
    params = list(itertools.product(Cs,Qs))
    
    models = [train(dtrain,C,Q) for C,Q in params]
    e_ins = [error(model,dtrain) for model in models]
    e_outs = [error(model,dtest) for model in models]
    nsvs = [len(model.support_) for model in models]

    print(list(zip(params,e_ins,e_outs,nsvs)))
    
    
    
def cv(folds,Cs):
    
    data = extract_one_vs_one(data_train,1,5)
    n = len(data)
    data = data[np.random.permutation(n)]
    def cverror(C):
        block = n//folds
        ret = 0
        for i in range(folds):
            selected = np.ones(n)==1
            selected[range(i*block,i*block+block)]=0
            dtrain = data[selected]       
            dval = data[~selected]
            model = train(dtrain,C,2)
            ret += error(model,dval)
        return ret/folds
    
    return [(C,cverror(C)) for C in Cs]
        
    
    