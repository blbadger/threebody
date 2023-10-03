import ctypes
import numpy as np

f = ctypes.CDLL('./library.so').save_array
f.restype = ctypes.POINTER(ctypes.c_float * 2**22)
arr = f().contents
print (type(arr))
np_arr = np.array(arr)
print (np_arr[:10])
