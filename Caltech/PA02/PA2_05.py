from numpy import *
from numpy.random import rand
from numpy.linalg import pinv

import matplotlib.pyplot as plt

def run_lr_once(N=100):
	pt_line = rand(4).reshape(2,2)*2-1
	pt_B = pt_line[0,:]
	pt_C = pt_line[1,:]
	def getY(pt_A):
		return sign(cross(pt_B-pt_A,pt_C-pt_B))
	def prepend_ones(Xs):
		N = Xs.shape[0]
		return hstack((ones(N).reshape(N,1),Xs))
	trainX = rand(N*2).reshape(N,2)*2-1
	testX = rand(N*20).reshape(N*10,2)*2-1
	trainY = getY(trainX)
	trainX_p = prepend_ones(trainX)
	testX_p = prepend_ones(testX)
	w = pinv(trainX_p).dot(trainY)

	result_in = sign(trainX_p.dot(w))
	result_out = sign(testX_p.dot(w))

	E_in = sum(result_in!=trainY) / N
	E_out = sum(result_out != getY(testX)) / N /10

	return array([E_in,E_out])


def run_pla_with_lr_once(N=10):
    ## PLA Algorithms
    ## return (#iterations,#learned,#real)
    real = rand(3)*2-1
    data = hstack([rand(N,2)*2-1, ones([N,1])])
    label = sign(sum(data*real, axis=1))
    current = pinv(data).dot(label)
    niter = 0

    def find_error(current):
        cur_label = sign(sum(data*current, axis=1))
        return [i for i in range(N) if cur_label[i]!=label[i]]

    while find_error(current):
        idx = random.choice(find_error(current))
        current += label[idx]*data[idx]
        niter+=1
    return niter,current,real





def run_lr_circle_once(N=100):
	def getY(pt):
		return sign(pt[:,0]**2+pt[:,1]**2-0.6)

	def addnoise(Ys):
		N = Ys.shape[0]
		return ((rand(N)<0.1)*-2 + 1)*Ys

	def transform(Xs):
		N = Xs.shape[0]
		#return hstack((ones(N).reshape(N,1),Xs))
		tocombine = [ones(N),Xs,Xs[:,0]*Xs[:,1],Xs[:,0]**2,Xs[:,1]**2]
		for i,item in enumerate(tocombine):
			if len(item.shape)==1:
				tocombine[i]=item.reshape(N,1)
		return hstack(tocombine)

	trainX = rand(N*2).reshape(N,2)*2-1
	testX = rand(N*20).reshape(N*10,2)*2-1
	trainY = addnoise(getY(trainX))
	testY = addnoise(getY(testX))
	trainX_p = transform(trainX)
	testX_p = transform(testX)
	w = pinv(trainX_p).dot(trainY)

	result_in = sign(trainX_p.dot(w))
	result_out = sign(testX_p.dot(w))

	#plt.scatter(trainX[:,0],trainX[:,1],marker='+',c=trainY,cmap=plt.cm.coolwarm,s=50)
	#plt.show()

	E_in = sum(result_in!=trainY) / N
	E_out = sum(result_out != testY) / N /10
	return array([E_in,E_out])

print(mean([run_lr_circle_once() for i in range(1000)] ,axis=0))