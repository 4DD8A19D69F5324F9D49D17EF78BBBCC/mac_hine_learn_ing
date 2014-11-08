# -*- coding: utf-8 -*-
from __future__ import division
from math import *

def mh(N,dvc):
    return N**dvc
def logmh(N,dvc):
    return dvc*log(N)    


def bisect_fun(f,lo,hi):
    while hi-lo>0.0001:
        mi=(lo+hi)/2
        tmp = f(mi)
        if tmp<mi:
            hi=mi
        else:
            lo=mi
    return mi
    

def getp_vc(N,dvc,eps):
    return 4*mh(2*N,dvc)*exp(-1.0/8*(eps**2)*N)
    

def geteps_vc(N,dvc,delta):
    return sqrt(8/N*(logmh(2*N,dvc)+log(4)-log(delta)))

def geteps_rpb(N,dvc,delta):
    return sqrt(2*(logmh(N,dvc)+log(2*N))/N) + sqrt(2/N*log(1/delta))+1/N


def geteps_pvb(N,dvc,delta):
    def helper(eps):
        return sqrt(1/N*(2*eps+(logmh(2*N,dvc)+log(6)-log(delta))))
    return bisect_fun(helper,0,10000)

def geteps_dev(N,dvc,delta):
    def helper(eps):
        return sqrt(1/2/N*(4*eps*(1+eps)+logmh(N**2,dvc)+log(4)-log(delta)))
    return bisect_fun(helper,0,10000)


def getresult(N,dvc,delta):
    funs = [geteps_vc,geteps_rpb,geteps_pvb,geteps_dev]
    return [f(N,dvc,delta) for f in funs]
    
    


