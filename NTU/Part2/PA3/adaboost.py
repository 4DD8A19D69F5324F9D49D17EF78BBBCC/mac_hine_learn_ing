# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 22:09:22 2015

@author: Administrator
"""

import numpy as np

def get_decision_stump(data,weight):
    N = len(data)
    D = len(data[0])-1
    def all_stumps():
        for i in range(D):
            xs_ = data[:,i]
            ys_ = data[:,-1]
            ws_ = weight[:]
            
            xs,ys,ws = zip(*sorted(zip(xs_,ys_,ws_)))
            xs = np.array(xs)
            ys = np.array(ys)
            ws = np.array(ws)
            
            total_w = sum(ws)
            current_neg = sum(ws[ys==-1])
            p = xs[0]-1

            def getitem(current_neg,p):
                err = min(current_neg,total_w - current_neg)
                s = 1 if err == current_neg else -1
                return (i,s,p,err)
            
            yield getitem(current_neg,p)
            for j in range(N):
                if ys[j] == -1:
                    current_neg -= ws[j]
                else:
                    current_neg += ws[j]
                
                if j!=N-1:                
                    p = (xs[j]+xs[j+1])/2.
                else:
                    p= xs[j]+1
                yield getitem(current_neg,p)
    return min(all_stumps(),key = lambda x:x[-1])
            

def test_stump(data,weight,stump):    
    d,s,theta,err = stump
    xs = data[:,d]
    ys = data[:,-1]
    ret = 0    
    for x,y,w in zip(xs,ys,weight):
        yp = s * np.sign(x-theta)
        ret += (yp!=y)* w
    return ret

def reweight(data,weight,stump):
    ret = weight.copy()
    d,s,theta,err = stump
    xs = data[:,d]
    ys = data[:,-1]
    yp = s*np.sign(xs-theta)
    
    sum_w = sum(weight)
    err_norm = err / sum_w
    delta = np.sqrt((1.-err_norm)/err_norm)  

    ret[ys!=yp]*=delta
    ret[ys==yp]/=delta
    return ret 
        
    
class AdaBoost(object):
    def __init__(self,data,niters = 300):
        self.stumps = []
        self.ws = []
        w  =np.array([1./len(data)]*len(data))
        for i in range(niters):
                     
            stump = get_decision_stump(data,w)
            err_norm = stump[-1]/sum(w)
            delta = np.sqrt((1.-err_norm)/err_norm)
            self.stumps.append(stump)
            self.ws.append(np.log(delta))
            w = reweight(data,w,stump)
            
    def predict(self,data):
        yp = np.zeros(len(data))        
        for stump,w in zip(self.stumps,self.ws):
            d,s,theta,err = stump
            yp += w * s * np.sign(data[:,d]-theta)
        return np.sign(yp)
        

data_train = np.loadtxt('hw2_adaboost_train.dat')
data_test = np.loadtxt('hw2_adaboost_test.dat')
init_w  =np.array([1./len(data_train)]*len(data_train))
stump =  get_decision_stump(data_train,init_w)

w2 = reweight(data_train, init_w,stump)
print stump
print test_stump(data_train,init_w,stump)
wr= reweight(data_train,init_w,stump)
clf= AdaBoost(data_train)
eout= sum( clf.predict(data_test[:,:-1]) !=  data_test[:,-1])*1.0 / len(data_test)




def kernel(a,b,gamma):
    return np.exp(-gamma* np.sum((a-b)**2))
def get_kernel_matrix(data,gamma):
    n = len(data)
    ret = np.zeros((n,n))
    xs = data[:,:-1]
    for i in range(n):
        for j in range(n):
            ret[i,j] = kernel(xs[i],xs[j],gamma)
    return ret

def get_beta(data,K,lambda_):
    n = len(data)
    return np.linalg.inv(lambda_*np.eye(n)+K).dot(data[:,-1])


betas =  get_beta(data_train,get_kernel_matrix(data_train,0.1),0.1)


dsvm = np.loadtxt('hw2_lssvm_all.dat')
dsvm_train = dsvm[:400,:]
dsvm_test = dsvm[400:,:]




for gamma in [32,2,0.125]:
    K = get_kernel_matrix(dsvm_train,gamma)
    dx = dsvm_train[:,:-1]
    dt = dsvm_test[:,:-1]

    for lambda_ in [0.001,1,1000]:
        beta = get_beta(dsvm_train,K,lambda_)
        
        def predict(x):
            return sum(b*kernel(d,x,gamma) for b,d in zip(beta,dx))+1e-5
            
        yp_train = np.sign([predict(x) for x in dx])
        yp_test = np.sign([predict(x) for x in dt])
        
        e_in = np.mean(yp_train != dsvm_train[:,-1])
        e_out = np.mean(yp_test != dsvm_test[:,-1])
        print gamma,lambda_,e_in,e_out
        
                
        
        
