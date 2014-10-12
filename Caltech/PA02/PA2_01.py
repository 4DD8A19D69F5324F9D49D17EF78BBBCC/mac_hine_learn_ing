from numpy import *
from numpy.random import *
def tryonce():
	return min(sum(randint(0,2,1000*10).reshape(1000,10),axis=1))/10

result = [ tryonce() for i in range(10000)]
print(sum(result)/len(result))
