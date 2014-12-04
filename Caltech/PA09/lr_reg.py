# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 18:01:30 2014

@author: Administrator
"""

import numpy as np
from sklearn import linear_model as lm
import itertools

data_train = np.loadtxt("features.train",dtype=float)
data_test = np.loadtxt("features.test",dtype=float)




def transform(data):
    Xs = data[:,:-1]
    Ys = data[:,-1]
    return np.column_stack([Xs,Xs[:,1]*Xs[:,2],Xs[:,1]**2,Xs[:,2]**2,Ys])

def extract_one_vs_all(data,x,transform=lambda x:x):
    Ys = data[:,0].copy()
    Xs = data[:,1:]
    p = (Ys == x)
    Ys[p]=1
    Ys[~p]=-1
    ret = np.column_stack([np.ones(len(Xs)),Xs,Ys])
    return transform(ret)


def extract_one_vs_one(data,x,y,transform=lambda x:x):
    Ys = data[:,0].copy()
    Xs = data[:,1:]
    
    p1 = (Ys == x)
    p2 = (Ys == y)
    Ys[p1]=1
    Ys[p2]=-1
    
    
    ret = np.column_stack([np.ones(np.sum(p1|p2)),Xs[p1|p2],Ys[p1|p2]])
    
    return transform(ret)
    
def train(data,**kwargs):
    clf = lm.RidgeClassifier(**kwargs)
    clf.fit(data[:,:-1],data[:,-1])
    return clf

def error(model,data):
    Yp = model.predict(data[:,:-1])
    Ys = data[:,-1]
    return np.mean(Yp!=Ys)



def problem_7():
    ns = [5,6,7,8,9]
    datas = [extract_one_vs_all(data_train,i) for i in ns]
    clfs = [train(data,alpha=1) for data in datas]
    errors = [ error(clf,data) for clf,data in zip(clfs,datas)]    
    print(errors)
    

def problem_8():
    ns = [0,1,2,3,4]
    datas = [extract_one_vs_all(data_train,i) for i in ns]
    datas_eval = [ extract_one_vs_all(data_test,i) for i in ns]
    clfs = [train(data,alpha=1) for data in datas]
    errors = [ error(clf,data) for clf,data in zip(clfs,datas_eval)]
    print(errors)     
        
def problem_9(transform=lambda x:x):
    ns = list(range(10))
    datas = [extract_one_vs_all(data_train,i,transform) for i in ns]
    datas_eval = [ extract_one_vs_all(data_test,i,transform) for i in ns]
    clfs = [train(data,alpha=1) for data in datas]
    errors = [ error(clf,data) for clf,data in zip(clfs,datas_eval)]
    print(list(zip(ns,errors)))      

def problem10():
    data_one = extract_one_vs_one(data_train,1,5,transform)
    data_eval = extract_one_vs_one(data_test,1,5,transform)
    alphas = [0.01,1]
    clfs = [train(data_one,alpha=alpha) for alpha in alphas]
    eins = [error(clf,data_one) for clf in clfs]
    eouts = [error(clf,data_eval) for clf in clfs]
    print(list(zip(alphas,eins)))
    print(list(zip(alphas,eouts)))
