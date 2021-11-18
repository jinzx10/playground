import numpy as np

a = np.random.rand(3,3)

filename = 'data.txt'

f = open(filename, 'w')
f.close()

f = open(filename, 'a')

for i in range(0, np.size(a,0)):
    f.write('Au ')
    np.savetxt(f, a[i,:], fmt='%17.12f', newline=" ")
    f.write('\n')
    

f.close()
