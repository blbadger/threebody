#! python3
# Produces trajectories of three bodies
# according to Netwon's gravitation: illustrates
# the three body problem where small changes to 
# initial conditions cause large changes to later
# positions

# import third-party libraries
import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import numexpr as ne 
import time
import torch
from queue import deque

device = 'cuda' if torch.cuda.is_available else 'cpu'
print (f'Device: {device}')

class Threebody:

	def __init__(self, time_steps, x_res, y_res, z_offset, m3=30, exponent=3):
		self.x_res = x_res
		self.y_res = y_res
		self.distance = 0.5
		self.m1 = 10
		self.m2 = 20
		self.m3 = m3
		self.time_steps = time_steps
		self.p1, self.p2, self.p3 = (torch.tensor([]) for i in range(3))
		self.v1, self.v2, self.v3 = (torch.tensor([]) for i in range(3))
		self.p1_prime, self.p2_prime, self.p3_prime = (torch.tensor([]) for i in range(3))
		self.v1_prime, self.v2_prime, self.v3_prime = (torch.tensor([]) for i in range(3))

		self.z_offset = z_offset

		# assign a small number to each time step
		self.delta_t = 0.001
		self.exponent = exponent


	def accelerations(self, p1, p2, p3):
		"""
		A function to calculate the derivatives of x, y, and z
		given 3 object and their locations according to Newton's laws

		Args:
			p1: np.ndarray(np.meshgrid[float]) or float
			p2: np.ndarray(np.meshgrid[float]) or float
			p3: np.ndarray(np.meshgrid[float]) or float

		Return:
			planet_1_dv: np.ndarray(np.meshgrid[float]) or float
			planet_2_dv: np.ndarray(np.meshgrid[float]) or float
			planet_3_dv: np.ndarray(np.meshgrid[float]) or float

		"""

		e = self.exponent
		m_1, m_2, m_3 = self.m1, self.m2, self.m3
		planet_1_dv = -9.8 * m_2 * (p1 - p2)/(torch.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)**e) - \
					   9.8 * m_3 * (p1 - p3)/(torch.sqrt((p1[0] - p3[0])**2 + (p1[1] - p3[1])**2 + (p1[2] - p3[2])**2)**e)

		planet_2_dv = -9.8 * m_3 * (p2 - p3)/(torch.sqrt((p2[0] - p3[0])**2 + (p2[1] - p3[1])**2 + (p2[2] - p3[2])**2)**e) - \
					   9.8 * m_1 * (p2 - p1)/(torch.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 + (p2[2] - p1[2])**2)**e)

		planet_3_dv = -9.8 * m_1 * (p3 - p1)/(torch.sqrt((p3[0] - p1[0])**2 + (p3[1] - p1[1])**2 + (p3[2] - p1[2])**2)**e) - \
					   9.8 * m_2 * (p3 - p2)/(torch.sqrt((p3[0] - p2[0])**2 + (p3[1] - p2[1])**2 + (p3[2] - p2[2])**2)**e)

		return planet_1_dv, planet_2_dv, planet_3_dv


	def not_diverged(self, p1, p1_prime):
		"""
		Find which trajectories have diverged from their shifted values

		Args:
			p1: np.ndarray[np.meshgrid[bool]]
			p1_prime: np.ndarray[np.meshgrid[bool]]

		Return:
			bool_arr: np.ndarray[bool]

		"""

		separation_arr = torch.sqrt((p1[0] - p1_prime[0])**2 + (p1[1] - p1_prime[1])**2 + (p1[2] - p1_prime[2])**2)
		bool_arr = separation_arr <= self.distance

		return bool_arr


	def plot_divergence(self, divergence_array):
		"""
		Generates a plot of a divergence array
		
		Args:
			divergence_array: np.ndarray[float]

		Returns:
			None (saves pyplot.imshow() image)

		"""
		
		plt.style.use('dark_background')
		plt.imshow(divergence_array, cmap='inferno')
		plt.axis('off')
		plt.savefig('Threebody_divergence{0:04d}.png'.format(i//100), bbox_inches='tight', pad_inches=0, dpi=420)
		plt.close()
		return

	def plot_projection(self, divergence_array, i):
		"""
		Generates a plot of a divergence array with the projection of a slope z = 12/10 * y

		Args:
			divergence_array: np.ndarray[float]

		Returns:
			None (saves pyplot.imshow() image)

		"""
		plt.rcParams.update({'font.size': 7})
		divergence_array[(self.p1[1] * 12 - self.p1[2] * 10 < 1).numpy() & (self.p1[1] * 12 - self.p1[2] * 10 > -1).numpy()] = i
		plt.style.use('dark_background')
		plt.imshow(divergence_array, cmap='inferno', extent=[-20, 20, -20, 20])
		plt.axis('on')
		plt.xlabel('y axis', fontsize=7)
		plt.ylabel('z axis', fontsize=7)
		plt.savefig('Threebody_divergence{0:04d}.png'.format(i//100), bbox_inches='tight', dpi=420)
		plt.close()
		return

	def initialize_arrays(self, xrange=0.002, yrange=0.002, x_center=10.5, y_center=0, double_type=True, shift=3e-6):
		"""
		Initialize torch.Tensor arrays

		kwargs:
			double_type: bool, if True then tensors are of type torch.float64 else float32

		returns:
			None

		"""
		y_lower, y_upper = y_center - yrange/2, y_center + yrange/2
		x_lower, x_upper = x_center - xrange/2, x_center + xrange/2
		y, x = np.arange(y_lower, y_upper, yrange/y_res), np.arange(x_lower, x_upper, xrange/x_res)
		grid = np.meshgrid(x, y)
		grid2 = np.meshgrid(x, y)

		# grid of all -11, identical starting z-values
		z_offset = self.z_offset
		z = np.zeros(grid[0].shape) + z_offset

		# shift the grid by a small amount
		grid2 = grid2[0] + shift, grid2[1] + shift
		# grid of all -11, identical starting z-values
		z_prime = np.zeros(grid[0].shape) - 11 + shift

		# p1_start = x_1, y_1, z_1
		p1 = np.array([grid[0], grid[1], z])
		p1_prime = np.array([grid2[0], grid2[1], z_prime])
		v1 = np.array([np.ones(grid[0].shape) * -3, np.zeros(grid[0].shape), np.zeros(grid[0].shape)])

		p2 = np.array([np.zeros(grid[0].shape), np.zeros(grid[0].shape), np.zeros(grid[0].shape)])
		v2 = np.array([np.zeros(grid[0].shape), np.zeros(grid[0].shape), np.zeros(grid[0].shape)])

		p3 = np.array([np.ones(grid[0].shape) * 10, np.ones(grid[0].shape) * 10, np.ones(grid[0].shape) * 12])
		v3 = np.array([np.ones(grid[0].shape) * 3, np.zeros(grid[0].shape), np.zeros(grid[0].shape)])

		# starting coordinates for planets shifted
		# p1_prime = np.array([x_prime, grid2[0], grid2[1]])
		v1_prime = np.array([np.ones(grid[0].shape) * -3, np.zeros(grid[0].shape), np.zeros(grid[0].shape)])

		p2_prime = np.array([np.zeros(grid[0].shape), np.zeros(grid[0].shape), np.zeros(grid[0].shape)])
		v2_prime = np.array([np.zeros(grid[0].shape), np.zeros(grid[0].shape), np.zeros(grid[0].shape)])

		p3_prime = np.array([np.ones(grid[0].shape) * 10, np.ones(grid[0].shape) * 10, np.ones(grid[0].shape) * 12])
		v3_prime = np.array([np.ones(grid[0].shape) * 3, np.zeros(grid[0].shape), np.zeros(grid[0].shape)])

		# convert numpy arrays to torch.Tensor([float64]) objects (2x speedup)
		p1, p2, p3 = torch.Tensor(p1), torch.Tensor(p2), torch.Tensor(p3)
		v1, v2, v3 = torch.Tensor(v1), torch.Tensor(v2), torch.Tensor(v3)
		p1_prime, p2_prime, p3_prime = torch.Tensor(p1_prime), torch.Tensor(p2_prime), torch.Tensor(p3_prime)
		v1_prime, v2_prime, v3_prime = torch.Tensor(v1_prime), torch.Tensor(v2_prime), torch.Tensor(v3_prime)

		if double_type:
			ttype = torch.double
		else:
			ttype = torch.float

		self.p1, self.p2, self.p3 = p1.type(ttype).to(device), p2.type(ttype).to(device), p3.type(ttype).to(device)
		self.v1, self.v2, self.v3 = v1.type(ttype).to(device), v2.type(ttype).to(device), v3.type(ttype).to(device)
		self.p1_prime, self.p2_prime, self.p3_prime = p1_prime.type(ttype).to(device), p2_prime.type(ttype).to(device), p3_prime.type(ttype).to(device)
		self.v1_prime, self.v2_prime, self.v3_prime = v1_prime.type(ttype).to(device), v2_prime.type(ttype).to(device), v3_prime.type(ttype).to(device)

		return

	def adams_bashforth(self, current, fn_arr, order=2):
		assert len(fn_arr) >= order
		if order == 4:
			# note that array is newest to the right, oldest left
			fn, fn_1, fn_2, fn_3 = fn_arr[-1], fn_arr[-2], fn_arr[-3], fn_arr[-4]
			v = current + (1/24) * self.delta_t * (55*fn - 59*fn_1 + 37*fn_2 - 9*fn_3)

		elif order == 2:
			# note that array is newest to the right, oldest left
			fn, fn_1 = fn_arr[-1], fn_arr[-2]
			v = current + (1/2) * self.delta_t * (3*fn - 1*fn_1)

		elif order == 1:
			v = current + self.delta_t * fn_arr[-1]

		return v

	def sensitivity_bashforth(self, iterations_video=False, double_type=True):
		"""
		Determine the sensitivity to initial values per starting point of planet 1, as
		measured by the time until divergence.

		kwargs:
			iterations_video: Bool, if True then divergence is plotted every 100 time steps

		Returns:
			time_array: np.ndarray[int] of iterations until divergence

		"""
		delta_t = self.delta_t
		self.initialize_arrays(double_type=double_type)
		time_array = torch.zeros(self.p1[0].shape).to(device)

		# bool array of all True
		still_together = time_array < 1e10
		dv1_arr, dv2_arr, dv3_arr = deque([]), deque([]), deque([])
		dv1_prime_arr, dv2_prime_arr, dv3_prime_arr = deque([]), deque([]), deque([])
		v1_arr, v2_arr, v3_arr = deque([]), deque([]), deque([])
		v1_prime_arr, v2_prime_arr, v3_prime_arr = deque([]), deque([]), deque([])

		t = time.time()
		# evolution of the system
		for i in range(self.time_steps):
			if i % 1000 == 0:
				print (f'Iteration: {i}')
				print (f'Completed in: {round(time.time() - t, 2)} seconds')
				t = time.time()
				time_array2 = i - time_array 
				if iterations_video:
					self.plot_projection(time_array2, i)

			not_diverged = self.not_diverged(self.p1, self.p1_prime)

			# points still together are not diverging now and have not previously
			still_together &= not_diverged

			# apply boolean mask to ndarray time_array
			time_array[still_together] += 1

			# calculate derivatives and store as class variables self.p1 ...
			dv1, dv2, dv3 = self.accelerations(self.p1, self.p2, self.p3)
			dv1_prime, dv2_prime, dv3_prime = self.accelerations(self.p1_prime, self.p2_prime, self.p3_prime)
			dv1_arr.append(dv1)
			dv2_arr.append(dv2)
			dv3_arr.append(dv3)
			dv1_prime_arr.append(dv1_prime)
			dv2_prime_arr.append(dv2_prime)
			dv3_prime_arr.append(dv3_prime)

			if i >= 4:
				nv1 = self.adams_bashforth(self.v1, dv1_arr)
				nv2 = self.adams_bashforth(self.v2, dv2_arr)
				nv3 = self.adams_bashforth(self.v3, dv3_arr)
				dv1_arr.popleft(), dv2_arr.popleft(), dv3_arr.popleft()
			else:
				nv1 = self.v1 + dv1 * delta_t
				nv2 = self.v2 + dv2 * delta_t
				nv3 = self.v3 + dv3 * delta_t

			if i >= 4:
				self.p1 = self.adams_bashforth(self.p1, v1_arr)
				self.p2 = self.adams_bashforth(self.p2, v2_arr)
				self.p3 = self.adams_bashforth(self.p3, v3_arr)
				v1_arr.popleft(), v2_arr.popleft(), v3_arr.popleft()
			else:
				self.p1 = self.p1 + self.v1 * delta_t
				self.p2 = self.p2 + self.v2 * delta_t
				self.p3 = self.p3 + self.v3 * delta_t

			self.v1, self.v2, self.v3 = nv1, nv2, nv3
			v1_arr.append(self.v1)
			v2_arr.append(self.v2)
			v3_arr.append(self.v3)

			if i >= 4:
				nv1_prime = self.adams_bashforth(self.v1_prime, dv1_prime_arr)
				nv2_prime = self.adams_bashforth(self.v2_prime, dv2_prime_arr)
				nv3_prime = self.adams_bashforth(self.v3_prime, dv3_prime_arr)

				dv1_prime_arr.popleft(), dv2_prime_arr.popleft(), dv3_prime_arr.popleft()
			else:
				nv1_prime = self.v1_prime + dv1_prime * delta_t
				nv2_prime = self.v2_prime + dv2_prime * delta_t
				nv3_prime = self.v3_prime + dv3_prime * delta_t

			if i >= 4:
				self.p1_prime = self.adams_bashforth(self.p1_prime, v1_prime_arr)
				self.p2_prime = self.adams_bashforth(self.p2_prime, v2_prime_arr)
				self.p3_prime = self.adams_bashforth(self.p3_prime, v3_prime_arr)
				v1_prime_arr.popleft(), v2_prime_arr.popleft(), v3_prime_arr.popleft()
			else:
				self.p1_prime = self.p1_prime + self.v1_prime * delta_t
				self.p2_prime = self.p2_prime + self.v2_prime * delta_t
				self.p3_prime = self.p3_prime + self.v3_prime * delta_t

			self.v1_prime, self.v2_prime, self.v3_prime = nv1_prime, nv2_prime, nv3_prime
			v1_prime_arr.append(self.v1_prime)
			v2_prime_arr.append(self.v2_prime)
			v3_prime_arr.append(self.v3_prime)

		return time_array

  
for i in range(1):
	time_steps = 50000
	x_res, y_res = 300, 300
	offset = -11
	mass = 30
	t = Threebody(time_steps, x_res, y_res, offset, mass)
	time_array = t.sensitivity_bashforth(iterations_video=False, double_type=True)
	time_array = time_steps - time_array
	time_array = time_array.cpu().numpy()
	plt.style.use('dark_background')
	plt.imshow(time_array, cmap='inferno')
	plt.axis('off')
	plt.savefig('Threebody_divergence{0:04d}.png'.format(i+1), bbox_inches='tight', pad_inches=0, dpi=410)
	plt.close()
 



 
