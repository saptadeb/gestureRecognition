import numpy as np
import random
a = np.zeros([3,4,3])
b = np.ones([3,2,3])
f = np.zeros([3,2,3])
#f2app=2

#print "a", a
#print "b",b


#print random.randint (1,100)

#a = a[:,:2,:]
c = np.append(a[:,:2,:],b,axis=1)
c = np.append(c,f,axis=1)

#c = np.append(a[:,1,:], b, axis=1)
#for i in range(b.shape[1]):
#	a[:,i+f2app,:] = b[:,i,:]

#print "a",a
#c = np.insert(a,b,axis=1)
print c