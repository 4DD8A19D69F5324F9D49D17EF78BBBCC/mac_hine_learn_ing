# -*- coding: utf-8 -*-
"""
Created on Thu Dec 04 21:03:34 2014

@author: Administrator
"""

import numpy as np
import matplotlib.pylab as plt
import matplotlib.pylab as pylab
from sklearn import svm

Xs = np.array([[1,0],[0,1],[0,-1],[-1,0],[0,2],[0,-2],[-2,0]])
Ys = np.array([-1,-1,-1,1,1,1,1])

Zs = np.column_stack([Xs[:,1]**2 - 2 * Xs[:,0]-1,Xs[:,0]**2-2*Xs[:,1]+1])


plt.scatter(x= Zs[:,0],y=Zs[:,1],c=Ys+1,cmap=pylab.cm.cool)

clf = svm.SVC(C=1e6,kernel='poly',degree=2)
clf.fit(Xs,Ys)
print(clf.support_)