import numpy as np
a = np.array([[1,2,3],[1,2,3]])
a = a.copy()
b = a[0,0:2]
b = b + 1
print(a)
print(b)