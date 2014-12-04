# -*- coding: utf-8 -*-

import numpy as np
from sklearn import svm, cluster, linear_model


def gen_data(N=100):
    Xs = np.random.rand(N,2)*2-1
    Ys = np.sign(Xs[:,1]-Xs[:,0]+0.25*np.sin(np.pi*Xs[:,0]))
    return np.column_stack([Xs,Ys])

class Lloyd_model(object):
    def __init__(self,gamma=1.5,K=9):
        self.gamma = gamma
        self.K = K
        
    def __transform__(self,Xs):
        cts = self.kmeans.cluster_centers_
        ret = np.column_stack([ np.exp(-self.gamma*np.sum((Xs-cts[i])**2,axis=1)) for i in range(len(cts))])
        return ret            
    def fit(self,Xs,Ys):        
        self.kmeans = cluster.KMeans(n_clusters=self.K)
        self.kmeans.fit(Xs)
        self.lm = linear_model.LinearRegression(fit_intercept=True)
        self.lm.fit(self.__transform__(Xs),Ys)
        
    def predict(self,Xs):
        pd = self.lm.predict(self.__transform__(Xs))>0
        return pd*2-1
        


def succ_rate(f,T=100):
    succ = 0.0    
    for i in range(T):
        succ += f()
    return succ/T

def error(model,data):
    return np.mean(model.predict(data[:,:-1]) != data[:,-1])

def problem_13(T=100):
    def run_once():
        data = gen_data()
        clf = svm.SVC(C=1e6,kernel='rbf',gamma=1.5)
        clf.fit(data[:,:-1],data[:,-1])
        return all(clf.predict(data[:,:-1]) == data[:,-1])
    return succ_rate(run_once,T)


def problem_14(T=100):
    def run_once():
        data = gen_data()
        clf = svm.SVC(C=1e6,kernel='rbf',gamma=1.5)
        clf2 = Lloyd_model(K=9)
        clf.fit(data[:,:-1],data[:,-1])
        clf2.fit(data[:,:-1],data[:,-1])
        dtest = gen_data(N=500)
        e1 = error(clf,dtest)
        e2 = error(clf2,dtest)
        return e1<e2
    return succ_rate(run_once,T)

def run_models(models):
    data = gen_data(N=100)
    for model in models:
        model.fit(data[:,:-1],data[:,-1])
    dtest=gen_data(N=1000)
    
    eins = [ error(model,data) for model in models]
    eouts = [ error(model,dtest) for model in models]
    return eins,eouts

print np.mean([run_models([Lloyd_model()])[0][0]==0  for i in range(100)])