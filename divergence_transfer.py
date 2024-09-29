import numpy as np
import ctypes
import matplotlib.pyplot as plt 

f = ctypes.CDLL('binaries/divergence.so').divergence
for i in range(300):
	x_res = 300
	y_res = 300
	time_steps = int(5000 + (350000 * i / 400))
	# time_steps = 50000
	shift_distance = 0.001 / (2**(i/30))
	x_center = 5.30031
	y_center = -0.45
	x_range = 40 / (2**(i/30))
	y_range = 40 / (2**(i/30))

	f.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double] # CUDA kernel arg types
	f.restype = ctypes.POINTER(ctypes.c_int * x_res * y_res) # kernel return type
	arr = f(x_res, y_res, time_steps, x_center, x_range, y_center, y_range, shift_distance).contents
	print (arr)
	time_array = np.array(arr)
	time_array = time_array.reshape(x_res, y_res)

	# print (time_array[:100])
	time_array = time_steps - time_array
	plt.style.use('dark_background')
	plt.imshow(time_array, cmap='inferno', aspect='auto')
	plt.axis('off')
	plt.savefig('Threebody_divergence{0:04d}.png'.format(i), bbox_inches='tight', pad_inches=0, dpi=410)
	plt.close()
	del arr

