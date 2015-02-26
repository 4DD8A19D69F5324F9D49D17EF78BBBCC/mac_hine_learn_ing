# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 09:54:41 2015

@author: Administrator
"""
import numpy as np
import random
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer
from pybrain.supervised.trainers import *
from pybrain.datasets import ClassificationDataSet
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import k_means
from collections import Counter

acti = np.tanh
acti_grad = lambda x: 1.0/(np.cosh(x)**2)
pad = lambda x: np.append([1],x)
unpad = lambda x:x[1:]

class NNet(object):
    def __init__(self,data,hidden,r=0.1):
        self.Xs = data[:,:-1]
        self.Ys = data[:,-1]        
        self.dim = np.shape(self.Xs)[1]
        self.hidden = hidden
        self.W0 = np.random.rand(self.dim+1,self.hidden) * r*2 - r
        self.W1 = np.random.rand(self.hidden+1,1) * r*2 -r
    
    def forward(self,x):
        self.S0 = pad(x).dot(self.W0)
        self.X0 = acti(self.S0)
        self.S1 = pad(self.X0).dot(self.W1)[0]
        self.X1 = acti(self.S1)
        return self.X1
        
    def backward(self,x,y):
        self.forward(x)
        self.D1 = -2*(y-self.X1) * acti_grad(self.S1)
        self.G1 = (self.D1 * pad(self.X0)).reshape(self.hidden+1,1)
        self.D0 = (nnet.D1 * unpad(nnet.W1)).T * acti_grad(nnet.S0)
        self.G0 = (self.D0.T*pad(x)).T
        return self.G0,self.G1
        
        
    def numerical_grad(self,x,y):
        R0 = np.zeros((self.dim+1,self.hidden))
        R1 = np.zeros((self.hidden+1,1))
        eps = 1e-4
        
        def compute(x):
            return (x-y)**2
        for i in range(self.dim+1):
            for j in range(self.hidden):
                self.W0[i,j] += eps
                fplus = compute(self.forward(x)) 
                self.W0[i,j] -= 2*eps
                fminus = compute(self.forward(x))
                self.W0[i,j] += eps
                R0[i,j] = (fplus - fminus)/2.0/eps
        
        
        for i in range(self.hidden+1):
            self.W1[i]+=eps
            fplus = compute(self.forward(x))
            self.W1[i] -= 2*eps
            fminus = compute(self.forward(x))
            self.W1[i] += eps
            R1[i] = (fplus - fminus)/2.0/eps

        return R0,R1
        
    def train(self,niter=50000,rate = 0.1):
        for i in range(niter):
            idx = random.randint(0,len(self.Xs)-1)
            self.backward(self.Xs[idx],self.Ys[idx])
            self.W0 -= rate*self.G0
            self.W1 -= rate*self.G1
    
    def predict(self,Xs):        
        Ys = [ self.forward(x) for x in Xs]
        return np.sign(Ys)
        
        
        
        
    
    
data_train = np.loadtxt('hw4_nnet_train.dat')
data_test = np.loadtxt('hw4_nnet_test.dat')
Ms = [1,6,11,16,21]
Rs = [0,0.001,0.1,10,1000]
Rates = [0.001,0.01,0.1,1,10]
errs = [0]*5
def run_nn():
    nnet = buildNetwork(2, 8, 3, 1, hiddenclass=TanhLayer, bias=True, outclass=TanhLayer)
    nnet.randomize()
    ds = [ClassificationDataSet(2,1) for i in range(len(data_train))]
    for row,d in zip(data_train,ds):
        d.addSample(tuple(row[:-1]),row[-1])
    
    trainer = BackpropTrainer(nnet,learningrate=0.01)
    
    for i in range(50000):
        idx = random.randint(0,len(data_train)-1)
        trainer.trainOnDataset(dataset=ds[idx])
    
    def run(nnet,data):
        return np.sign([nnet.activate(x[:-1])[0] for x in data])
    return np.mean(run(nnet,data_test)!= data_test[:,-1])

data_train = np.loadtxt('hw4_knn_train.dat')
data_test = np.loadtxt('hw4_knn_test.dat')

class KNN(object):
    def __init__(self,k):
        self.knn = NearestNeighbors(n_neighbors=k)
        
    def fit(self,Xs,Ys):
        self.knn.fit(Xs)
        self.Ys = Ys
    
    def predict(self,Xs):
        distances, indices = self.knn.kneighbors(Xs)
        
        def get_result(idxs):
            ys =[self.Ys[idx] for idx in idxs]
            return Counter(ys).most_common(1)[0][0]
        return np.array([get_result(idxs) for idxs in indices])


knn = KNN(5)
knn.fit(data_train[:,:-1],data_train[:,-1])
print np.mean(knn.predict(data_train[:,:-1]) != data_train[:,-1])
print np.mean(knn.predict(data_test[:,:-1]) != data_test[:,-1])


data_train = np.loadtxt('hw4_kmeans_train.dat')

pt,assignment,inertia = k_means(data_train,10)
err = 0
for i,a in enumerate(assignment):
    pa = pt[a]
    now = data_train[i]
    err += np.linalg.norm(now-pa,2)**2
print 'KMeans Err=',err/len(data_train)