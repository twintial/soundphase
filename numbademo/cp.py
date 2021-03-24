import cupy as cp
import numpy as np

a = cp.array([1,2,3])
b = np.array([1,2,3])
print(a * cp.asarray(b))