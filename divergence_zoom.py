import numpy as np
import ctypes
import matplotlib.pyplot as plt 


def plot(output_array, index):
	plt.style.use('dark_background')
	plt.imshow(output_array, cmap='inferno', aspect='auto')
	plt.axis('off')
	plt.savefig('Threebody_divergence{0:04d}.png'.format(index), bbox_inches='tight', pad_inches=0, dpi=410)
	plt.close()
	return


f = ctypes.CDLL('./divergence_zoom.so').divergence
already_computed = {}
last_time_steps = 0
for i in range(60):
	x_res = 300
	y_res = 300
	time_steps = int(50000 + (350000 * i / 400))
	for pair in already_computed:
		already_computed[pair] += time_steps - last_time_steps
	last_time_steps = time_steps

	shift_distance = 0.001 / (2**(i/30))
	x_center = 5.30031
	y_center = -0.45
	x_range = 40 / (2**(i/30))
	y_range = 40 / (2**(i/30))
	x, y = [], []
	return_template = []
	decimal = int(-np.log(x_range / (x_res)))
	decimal = 2

	start_x = x_center - x_range/2
	start_y = y_center - y_range/2
	for j in range(int(x_res*y_res)):
		remainder = j % y_res
		step = j // x_res
		x_i = start_x + x_range*(remainder/x_res)
		y_i = start_y + y_range*(step/y_res)

		if (round(x_i, decimal), round(y_i, decimal)) not in already_computed:
			x.append(x_i) 
			y.append(y_i)
			return_template.append(-1)
		else:
			return_template.append(already_computed[(round(x_i, decimal), round(y_i, decimal))])
	length_x, length_y = len(x), len(y)
	print (f'N elements: {length_x}')

	x_array_type = ctypes.c_float * len(x)
	y_array_type = ctypes.c_float * len(y)
	x = x_array_type(*x)
	y = y_array_type(*y)

	f.argtypes = [ctypes.c_int, 
		ctypes.c_int, 
		ctypes.c_int, 
		ctypes.c_double, 
		ctypes.c_double, 
		ctypes.c_double, 
		ctypes.c_double, 
		ctypes.c_double, 
		ctypes.POINTER(ctypes.c_float*len(x)), 
		ctypes.POINTER(ctypes.c_float*len(y)),
		ctypes.c_int
		] 

	f.restype = ctypes.POINTER(ctypes.c_int * length_x) # kernal return type
	arr = f(x_res, y_res, time_steps, x_center, x_range, y_center, y_range, shift_distance, x, y, length_x).contents
	time_array = np.array(arr)
	flattened_arr = time_array.flatten()
	return_arr = []
	inc = 0
	for k in range(len(return_template)):
		if return_template[k] == -1:
			return_arr.append(flattened_arr[inc])
			already_computed[(round(x[inc], decimal), round(y[inc], decimal))] = flattened_arr[inc]
			inc += 1
		else:
			return_arr.append(return_template[k])


	output_array = np.array(return_arr).reshape(x_res, y_res)
	output_array = time_steps - output_array
	plot(output_array, i)

	# clean computed map
	keys_to_delete = []
	for pair in already_computed.keys():
		if pair[0] > start_x + x_range or pair[0] < start_x or pair[1] > start_y + y_range or pair[1] < start_y:
			keys_to_delete.append(pair)
		npair = []
		npair[0] = round(pair[0], decimal)
		npair[1] = round(already[0], decimal)

	print (len(already_computed))
	for k in keys_to_delete:
		already_computed.pop(k, None)
	print (len(already_computed))
