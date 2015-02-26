# -*- coding: utf-8 -*-
"""
Created on Mon Feb 09 09:05:01 2015

@author: Administrator
"""

import numpy as np
from collections import Counter

def get_decision_stump(data):
    N = len(data)
    D = len(data[0])-1
    
    weight = np.array([1]*len(data))
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



def get_gini_stump(data):
    N = len(data)
    D = len(data[0])-1
    
    weight = np.array([1./N]*len(data))
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
                
                err = 1-(current_neg**2 + (total_w - current_neg)**2)
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



class DecisionTree(object):
    def __init__(self,value,isleaf):
        self.value = value
        self.isleaf = isleaf
        self.childs = []
        
        self.size = 1
        self.leaf = isleaf
        
    def attach_left(self,left):
        self.left = left
        self.size += left.size
        self.leaf += left.leaf
        
    def attach_right(self,right):
        self.right = right
        self.size += right.size
        self.leaf += right.leaf
        
    def predict(self,x):   
        if self.isleaf:
            return self.value
        else:
            if self.predicate(x)==-1:
                return self.left.predict(x)
            else:
                return self.right.predict(x)
    
    def predict_all(self,xs):
        return np.array([self.predict(x) for x in xs])
            
    def predicate(self,data):
        d,s,theta,err = self.value
        xs = data[d]
        yp = s*np.sign(xs-theta)
        return yp
        
    def predicate_all(self,data):
        d,s,theta,err = self.value
        xs = data[:,d]
        yp = s*np.sign(xs-theta)
        return yp
    
    
    def __str__(self):
        return 'DecisionTree(%d nodes, %d leafs)'%(self.size,self.leaf)
        
def train_desicion_tree(data):
    ys = data[:,-1]
    if len(Counter(ys))==1:
        return DecisionTree(ys[0],True)
    else:
        stump = get_gini_stump(data)
        tree = DecisionTree(stump,False)
        yp = tree.predicate_all(data)

        if sum(yp==-1)!=0 and sum(yp==1)!=0:
            tree.attach_left(train_desicion_tree(data[yp==-1]))
            tree.attach_right(train_desicion_tree(data[yp==1]))
            return tree
        else:
            return DecisionTree(Counter(ys).most_common()[0][0],True)

def train_desicion_stump(data,leaf=False):
    ys = data[:,-1]
    if len(Counter(ys))==1:
        return DecisionTree(ys[0],True)
    else:
        stump = get_decision_stump(data)
        tree = DecisionTree(stump,False)
        yp = tree.predicate_all(data)

        if sum(yp==-1)!=0 and sum(yp==1)!=0 and (not leaf):
            tree.attach_left(train_desicion_stump(data[yp==-1],True))
            tree.attach_right(train_desicion_stump(data[yp==1],True))
            return tree
        else:
            return DecisionTree(Counter(ys).most_common()[0][0],True)

data_train = np.loadtxt("hw3_train.dat")
data_test = np.loadtxt("hw3_test.dat")



def ein(clf):
    return np.mean(clf.predict_all(data_train) != data_train[:,-1])

def eout(clf):
    return np.mean(clf.predict_all(data_test) != data_test[:,-1])

def avg_ein(clfs):
    return np.mean(map(ein,clfs))

def avg_eout(clfs):
    return np.mean(map(eout,clfs))

cart = train_desicion_tree(data_train)
print 'Q13',ein(cart)
print 'Q14',eout(cart)




def bootstrap(data):
    n = len(data)
    return data[np.random.choice(np.arange(n),n)]

class RandomForest(object):
    def __init__(self,ntree=300):
        self.ntree = ntree
        self.trees = []        
        
    def train(self,data):        
        for i in range(self.ntree):
           bdata =bootstrap(data)
           self.trees.append(train_desicion_tree(bdata))
    
    def predict(self,x):
        pd = [t.predict(x) for t in self.trees]
        return Counter(pd).most_common()[0][0]
    
    def predict_all(self,xs):
        return np.array([self.predict(x) for x in xs])

    def ein_t(self):
        return avg_ein(self.trees)

class RandomStump(object):
    def __init__(self,ntree=300):
        self.ntree = ntree
        self.trees = []        
        
    def train(self,data):        
        for i in range(self.ntree):
           bdata =bootstrap(data)
           self.trees.append(train_desicion_stump(bdata))
    
    def predict(self,x):
        pd = [t.predict(x) for t in self.trees]
        return Counter(pd).most_common()[0][0]
    
    def predict_all(self,xs):
        return np.array([self.predict(x) for x in xs])

    def ein_t(self):
        return avg_ein(self.trees)

#rfs = []
#for i in range(300):
#    rf = RandomForest(ntree=300)
#    rf.train(data_train)
#    rfs.append(rf)
#    print i

#rss = []
#for i in range(300):
#    rs = RandomStump(ntree=300)
#    rs.train(data_train)
#    rss.append(rs)
#    print i

