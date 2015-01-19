# -*- coding: utf-8 -*-
"""
Created on Sun Jan 11 09:52:07 2015

@author: Administrator
"""


import numpy as np
import sklearn.svm as svm
import sklearn.linear_model as lm
import cvxpy as cvx
data = np.array([
        [1,0,-1],
        [0,1,-1],
        [0,-1,-1],
        [-1,0,1],
        [0,2,1],
        [0,-2,1],
        [-2,0,1]
        ])
        

def transform(data):
    X1 = data[:,0]
    X2 = data[:,1]
    Y = data[:,2]
    return np.column_stack([X2**2-2*X1+3,X1**2-2*X2-3,Y])

def q2():
    lsvc = svm.LinearSVC(C=10**7,dual=False)
    tdata = transform(data)    
    lsvc.fit(tdata[:,:-1],tdata[:,-1])
    print lsvc.coef_
    print lsvc.intercept_

def q3():
    def K(x1,x2):
        return (1+ np.inner(x1[:-1],x2[:-1]))**2
    
    Q = np.zeros((len(data),len(data)))
    
    for i in range(len(data)):
        for j in range(len(data)):
            Q[i,j]= data[i,-1]*data[j,-1]*K(data[i],data[j])
    alpha = cvx.Variable(len(data))
    
    obj = cvx.Minimize(0.5*cvx.quad_form(alpha,Q)-cvx.sum_entries(alpha))
    constraint = [cvx.sum_entries(cvx.mul_elemwise(data[:,-1],alpha)) == 0, alpha>=0]
    prob = cvx.Problem(obj,constraint)
    prob.solve()
    ret = alpha.value
    
    svid = next(i for i,x in enumerate(ret) if x>1e-5)    
    
    b = data[svid,-1] - sum(ret[i,0]*data[i,-1]*K(data[i],data[svid]) for i in range(len(data)) if ret[i,0]>1e-5)
    
    def getvalue(X):
        XX = np.append(X,[-1])
        return sum(ret[i,0]*data[i,-1]*K(XX,data[i]) for i in range(len(data))) + b
    
    ks = [(i,j,i**2,j**2) for j in range(10) for i in range(10)]
    vs = [getvalue(k[:2]) for k in ks]
    
    lr = lm.LinearRegression()
    lr.fit(ks,vs)
    print ret
    print lr.coef_*9,lr.intercept_*9

q3()