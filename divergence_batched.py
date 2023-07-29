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

device = 'cuda' if torch.cuda.is_available else 'cpu'
print (f'Device: {device}')

class Threebody:

	def __init__(self, time_steps, x_res, y_res, z_offset, masses, batch=8):
		self.x_res = x_res
		self.y_res = y_res
		self.distance = 0.5
		self.m1 = 10
		self.m2 = 20
		self.m3 = masses
		self.time_steps = time_steps
		self.p1, self.p2, self.p3 = (torch.tensor([]) for i in range(3))
		self.v1, self.v2, self.v3 = (torch.tensor([]) for i in range(3))
		self.p1_prime, self.p2_prime, self.p3_prime = (torch.tensor([]) for i in range(3))
		self.v1_prime, self.v2_prime, self.v3_prime = (torch.tensor([]) for i in range(3))

		self.z_offset = z_offset

		# assign a small number to each time step
		self.delta_t = 0.001
		self.batch = batch


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

		m_1, m_2, m_3 = self.m1, self.m2, self.m3
		planet_1_dv = -9.8 * m_2 * (p1 - p2)/(torch.sqrt((p1[:, 0] - p2[:, 0])**2 + (p1[:, 1] - p2[:, 1])**2 + (p1[:, 2] - p2[:, 2])**2)**3).unsqueeze(1) - \
					   9.8 * m_3 * (p1 - p3)/(torch.sqrt((p1[:, 0] - p3[:, 0])**2 + (p1[:, 1] - p3[:, 1])**2 + (p1[:, 2] - p3[:, 2])**2)**3).unsqueeze(1)

		planet_2_dv = -9.8 * m_3 * (p2 - p3)/(torch.sqrt((p2[:, 0] - p3[:, 0])**2 + (p2[:, 1] - p3[:, 1])**2 + (p2[:, 2] - p3[:, 2])**2)**3).unsqueeze(1) - \
					   9.8 * m_1 * (p2 - p1)/(torch.sqrt((p2[:, 0] - p1[:, 0])**2 + (p2[:, 1] - p1[:, 1])**2 + (p2[:, 2] - p1[:, 2])**2)**3).unsqueeze(1)

		planet_3_dv = -9.8 * m_1 * (p3 - p1)/(torch.sqrt((p3[:, 0] - p1[:, 0])**2 + (p3[:, 1] - p1[:, 1])**2 + (p3[:, 2] - p1[:, 2])**2)**3).unsqueeze(1) - \
					   9.8 * m_2 * (p3 - p2)/(torch.sqrt((p3[:, 0] - p2[:, 0])**2 + (p3[:, 1] - p2[:, 1])**2 + (p3[:, 2] - p2[:, 2])**2)**3).unsqueeze(1)

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
		separation_arr = torch.sqrt((p1[:, 0] - p1_prime[:, 0])**2 + (p1[:, 1] - p1_prime[:, 1])**2 + (p1[:, 2] - p1_prime[:, 2])**2)
		bool_arr = separation_arr <= self.distance

		return bool_arr


	def initialize_arrays(self, double_type=True):
		"""
		Initialize torch.Tensor arrays

		kwargs:
			double_type: bool, if True then tensors are of type torch.float64 else float32

		returns:
			None

		"""
		batch = self.batch
		y, x = np.arange(-20, 20, 40/y_res), np.arange(-20, 20, 40/x_res)
		grid = np.meshgrid(x, y)
		grid2 = np.meshgrid(x, y)

		# grid of all -11, identical starting z-values
		z_offset = self.z_offset
		z = np.zeros(grid[0].shape) + z_offset # - 11

		# shift the grid by a small amount
		grid2 = grid2[0] + 1e-3, grid2[1] + 1e-3
		# grid of all -11, identical starting z-values
		z_prime = np.zeros(grid[0].shape) - 11 + 1e-3

		# p1_start = x_1, y_1, z_1
		p1 = np.array([grid[0], grid[1], z])
		p1_prime = np.array([grid2[0], grid2[1], z_prime])

		# settings for yz array
		# z, y = np.arange(20, -20, -40/self.y_res), np.arange(-20, 20, 40/self.x_res)
		# grid = np.meshgrid(y, z)
		# grid2 = np.meshgrid(y, z)

		# # grid of all -10, identical starting x-values
		# x = np.zeros(grid[0].shape) - 10

		# # shift the grid by a small amount
		# grid2 = grid2[0] + 1e-3, grid2[1] + 1e-3

		# # grid of all -10, identical starting x-values
		# x_prime = np.zeros(grid[0].shape) - 10 + 1e-3

		# starting coordinates for planets
		# p1 = np.array([x, grid[0], grid[1]])
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

		# points are shape [batch, planet, x, y]
		self.p1, self.p2, self.p3 = self.p1.repeat(batch, 1, 1, 1), self.p2.repeat(batch, 1, 1, 1), self.p3.repeat(batch, 1, 1, 1)
		self.v1, self.v2, self.v3 = self.v1.repeat(batch, 1, 1, 1), self.v2.repeat(batch, 1, 1, 1), self.v3.repeat(batch, 1, 1, 1)
		self.p1_prime, self.p2_prime, self.p3_prime = self.p1_prime.repeat(batch, 1, 1, 1), self.p2_prime.repeat(batch, 1, 1, 1), self.p3_prime.repeat(batch, 1, 1, 1)
		self.v1_prime, self.v2_prime, self.v3_prime = self.v1_prime.repeat(batch, 1, 1, 1), self.v2_prime.repeat(batch, 1, 1, 1), self.v3_prime.repeat(batch, 1, 1, 1)

		return


	def sensitivity(self, iterations_video=False):
		"""
		Determine the sensitivity to initial values per starting point of planet 1, as
		measured by the time until divergence.

		kwargs:
			iterations_video: Bool, if True then divergence is plotted every 100 time steps

		Returns:
			time_array: np.ndarray[int] of iterations until divergence

		"""

		delta_t = self.delta_t
		self.initialize_arrays(double_type=True)
		time_array = torch.zeros(self.p1[:, 0].shape).to(device)

		# bool array of all True
		still_together = torch.ones(time_array.shape, dtype=torch.bool).to(device)

		t = time.time()
		# evolution of the system
		for i in range(self.time_steps):
			if i % 1000 == 0:
				print (f'Iteration: {i}')
				print (f'Comleted in: {round(time.time() - t, 2)} seconds')
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

			nv1 = self.v1 + dv1 * delta_t
			nv2 = self.v2 + dv2 * delta_t
			nv3 = self.v3 + dv3 * delta_t

			self.p1 = self.p1 + self.v1 * delta_t
			self.p2 = self.p2 + self.v2 * delta_t
			self.p3 = self.p3 + self.v3 * delta_t
			self.v1, self.v2, self.v3 = nv1, nv2, nv3

			nv1_prime = self.v1_prime + dv1_prime * delta_t
			nv2_prime = self.v2_prime + dv2_prime * delta_t
			nv3_prime = self.v3_prime + dv3_prime * delta_t

			self.p1_prime = self.p1_prime + self.v1_prime * delta_t
			self.p2_prime = self.p2_prime + self.v2_prime * delta_t
			self.p3_prime = self.p3_prime + self.v3_prime * delta_t
			self.v1_prime, self.v2_prime, self.v3_prime = nv1_prime, nv2_prime, nv3_prime

		return time_array


batch = 8
for i in range(339, 1000, batch):
	time_steps = 50000
	x_res, y_res = 1000, 1000
	offset = -11 
	masses = torch.tensor([30 - j / 20  for j in range(i, i+batch)]).unsqueeze(1).unsqueeze(1).unsqueeze(1)
	masses = masses.repeat(1, 3, x_res, y_res).to(device)

	t = Threebody(time_steps, x_res, y_res, offset, masses, batch=batch)
	time_array = t.sensitivity(iterations_video=False)
	print (time_array.shape)
	for b in range(batch):
		t_arr = time_array[b, :, :]
		print (t_arr.shape)
		t_arr = time_steps - t_arr
		t_arr = t_arr.cpu().numpy()
		plt.style.use('dark_background')
		plt.imshow(t_arr, cmap='inferno')
		plt.axis('off')
		plt.savefig('Threebody_divergence{0:04d}.png'.format(i+b), bbox_inches='tight', pad_inches=0, dpi=410)
 




