# -*- coding: utf-8 -*-
"""
Created on Sun Nov  2 10:15:20 2014

@author: Administrator
"""


import numpy as np
from math import exp

def e_in(sigma,d,N):
    return sigma**2*(1- (d+1)/N)
    
def f(u,v):
    return (u*exp(v)-2*v*exp(-u))**2

def du(u,v):
    return 2*(u*exp(v)-2*v*exp(-u))*(exp(v)+2*v*exp(-u))

def dv(u,v):
    return 2*(u*exp(v)-2*v*exp(-u))*(u*exp(v)-2*exp(-u))
    
    
def grad(u,v):
    return np.array([du(u,v),dv(u,v)])


pt = np.array([1.,1.],dtype='float64')
rate = 0.1
for i in range(1,100):
    pt[0]-= rate*du(pt[0],pt[1])
    pt[1]-= rate*dv(pt[0],pt[1])    
    print("Iter ",i,pt,f(pt[0],pt[1]))

