import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

import time

key = random.PRNGKey(0)
x = random.normal(key, (10,))
print(x)

size = 3000
x = random.normal(key, (size, size), dtype=jnp.float32)

start = time.time()
for i in range(100):
    jnp.dot(x, x.T).block_until_ready()  # runs on the GPU
end = time.time()
print("time spent:", end - start)



import numpy as np
x = np.random.normal(size=(size, size)).astype(np.float32)
start = time.time()
for i in range(100):
    jnp.dot(x, x.T).block_until_ready()
end = time.time()
print("time spent:", end - start)

from jax import device_put
x = np.random.normal(size=(size, size)).astype(np.float32)
x = device_put(x)
start = time.time()
for i in range(100):
    jnp.dot(x, x.T).block_until_ready()
end = time.time()
print("time spent:", end - start)

