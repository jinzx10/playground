from mpi4py import MPI
import jax
import jax.numpy as jnp
import mpi4jax
import time
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def foo(arr):
   arr = arr + rank
   arr_sum, _ = mpi4jax.allreduce(arr, op=MPI.SUM, comm=comm)
   return arr_sum

foo_cpu = jax.jit(foo, backend='cpu')
#foo_cpu = foo


a = jnp.zeros((3, 3))

#start = time.time()
#result = foo_cpu(a)
#end = time.time()
#print("Time taken for CPU: ", end - start)

start = time.time()
result = foo_cpu(b)
end = time.time()
print("Time taken for CPU: ", end - start)

if rank == 0:
   print(result)
