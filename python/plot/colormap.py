import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

#x,y,c = zip(*np.random.rand(30,3)*4-2)
x = np.random.rand(10)
y = np.random.rand(10)
c = np.random.rand(10)

norm=plt.Normalize(-1,1)
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","violet","blue"])

plt.scatter(x,y,c=c, cmap=cmap, norm=norm)
plt.colorbar()
plt.show()




