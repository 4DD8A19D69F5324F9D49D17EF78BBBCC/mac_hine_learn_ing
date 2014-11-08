# -*- coding: utf-8 -*-

import numpy as np
from scipy import integrate


def err_fun(x,k=0.78):
    return (k*x-sin(np.pi*x))**2

def err_var(x,k=0.78):
    return (k*x-0.78*x)**2

def getks(ntries):
    xs = (np.random.rand(ntries)*2-1).reshape(ntries/2,2)
    ys = np.sin(np.pi*xs)
    ks = (ys[:,1]+ys[:,0])/(xs[:,1]+xs[:,0])
    return ks



    
