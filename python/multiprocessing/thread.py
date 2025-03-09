import numpy as np

def foo(bar):
  print('hello {0}'.format(bar))
  return 'foo' + str(bar)

from multiprocessing.pool import ThreadPool
pool = ThreadPool(processes=8)

for result in pool.map(foo, range(10)):
  print(result)


#async_result = pool.apply_async(foo, ('world', 'foo')) # tuple of args for foo

# do some other stuff in the main process

#return_val = async_result.get()  # get the return value from your function.
