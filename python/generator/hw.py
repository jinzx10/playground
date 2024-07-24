from itertools import chain
import numpy as np

def gen(n):
    for i in range(n):
        yield (i, i*2)

#for i, x in enumerate(gen(5)):
#    print(i, x)

def f(n):
    return n

#l, l2 = zip(*[(f(i), f(2*i)) for i in range(5)])
l, l2 = zip((f(i), f(2*i)) for i in range(5))


print(l)
print(l2)

