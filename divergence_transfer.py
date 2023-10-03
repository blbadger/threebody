import numpy as np
import ctypes
import matplotlib.pyplot as plt 

f = ctypes.CDLL('./divergence.so').divergence
dim = 1000
time_steps = 200000
f.restype = ctypes.POINTER(ctypes.c_int * dim**2)
arr = f(dim, time_steps).contents
print (type(arr))
time_array = np.array(arr)
time_array = time_array.reshape(dim, dim)

print (time_array[:100])
time_array = time_steps - time_array
plt.style.use('dark_background')
plt.imshow(time_array, cmap='inferno')
plt.axis('off')
plt.savefig('Threebody_divergence_cuda.png', bbox_inches='tight', pad_inches=0, dpi=410)
plt.close()
