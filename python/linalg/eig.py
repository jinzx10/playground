import numpy as np

num = 10
sz = 2
count = 0

def issorted(l):
    return all(l[i] <= l[i+1] for i in range(len(l)-1))

for i in range(0,num):
    a = np.random.rand(sz,sz)
    a = a + a.transpose()
    [val,vec] = np.linalg.eig(a)
    #a = a + (val[1]-val[0]+1)*(vec[:,0]*vec[:,0].transpose())
    val = np.linalg.eig(a)[0]
    print(val)
    print(a)
    count = count + issorted(val)

print(count)
