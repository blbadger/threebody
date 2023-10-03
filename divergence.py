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

	def __init__(self, time_steps, x_res, y_res, z_offset, m3=30):
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
		planet_1_dv = -9.8 * m_2 * (p1 - p2)/(torch.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)**3) - \
					   9.8 * m_3 * (p1 - p3)/(torch.sqrt((p1[0] - p3[0])**2 + (p1[1] - p3[1])**2 + (p1[2] - p3[2])**2)**3)

		planet_2_dv = -9.8 * m_3 * (p2 - p3)/(torch.sqrt((p2[0] - p3[0])**2 + (p2[1] - p3[1])**2 + (p2[2] - p3[2])**2)**3) - \
					   9.8 * m_1 * (p2 - p1)/(torch.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 + (p2[2] - p1[2])**2)**3)

		planet_3_dv = -9.8 * m_1 * (p3 - p1)/(torch.sqrt((p3[0] - p1[0])**2 + (p3[1] - p1[1])**2 + (p3[2] - p1[2])**2)**3) - \
					   9.8 * m_2 * (p3 - p2)/(torch.sqrt((p3[0] - p2[0])**2 + (p3[1] - p2[1])**2 + (p3[2] - p2[2])**2)**3)

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

	def initialize_arrays(self, double_type=True):
		"""
		Initialize torch.Tensor arrays

		kwargs:
			double_type: bool, if True then tensors are of type torch.float64 else float32

		returns:
			None

		"""
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

		return


	def sensitivity(self, iterations_video=False, double_type=True):
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

		print (self.p1, self.p1_prime)
		return time_array


	def three_body_phase(self):
		"""
		Plot the phase of three bodies according to Newtonian mechanics

		Args:
			None

		Returns:
			None (saves matplotlib.pyplot object image)

		"""

		# starting coordinates for planets
		# p1_start = x_1, y_1, z_1
		p1_start = np.array([-10, 10, -11])
		v1_start = np.array([-3, 0, 0])

		# p2_start = x_2, y_2, z_2
		p2_start = np.array([0, 0, 0])
		v2_start = np.array([0, 0, 0])

		# p3_start = x_3, y_3, z_3
		p3_start = np.array([10, 10, 12])
		v3_start = np.array([3, 0, 0])

		# starting coordinates for planets shifted
		# p1_start = x_1, y_1, z_1
		p1_start_prime = np.array([-9.999, 10.001, -10.999])
		v1_start_prime = np.array([-3, 0, 0])

		# p2_start = x_2, y_2, z_2
		p2_start_prime = np.array([0, 0, 0])
		v2_start_prime = np.array([0, 0, 0])

		# p3_start = x_3, y_3, z_3
		p3_start_prime = np.array([10, 10, 12])
		v3_start_prime = np.array([3, 0, 0])

		# parameters
		delta_t = self.delta_t
		steps = self.time_steps

		# initialize solution array
		p1 = np.array([[0.,0.,0.] for i in range(steps)])
		v1 = np.array([[0.,0.,0.] for i in range(steps)])

		p2 = np.array([[0.,0.,0.] for j in range(steps)])
		v2 = np.array([[0.,0.,0.] for j in range(steps)])

		p3 = np.array([[0.,0.,0.] for k in range(steps)])
		v3 = np.array([[0.,0.,0.] for k in range(steps)])


		p1_prime = np.array([[0.,0.,0.] for i in range(steps)])
		v1_prime = np.array([[0.,0.,0.] for i in range(steps)])

		p2_prime = np.array([[0.,0.,0.] for j in range(steps)])
		v2_prime = np.array([[0.,0.,0.] for j in range(steps)])

		p3_prime = np.array([[0.,0.,0.] for k in range(steps)])
		v3_prime = np.array([[0.,0.,0.] for k in range(steps)])

		# starting point
		p1[0], p2[0], p3[0] = p1_start, p2_start, p3_start
		v1[0], v2[0], v3[0] = v1_start, v2_start, v3_start

		p1_prime[0], p2_prime[0], p3_prime[0] = p1_start_prime, p2_start_prime, p3_start_prime
		v1_prime[0], v2_prime[0], v3_prime[0] = v1_start_prime, v2_start_prime, v3_start_prime
		time = [0]

		# evolution of the system
		for i in range(steps-1):
			time.append(i)
			#calculate derivatives
			dv1, dv2, dv3 = self.optimized_accelerations(p1[i], p2[i], p3[i])
			dv1_prime, dv2_prime, dv3_prime = self.optimized_accelerations(p1_prime[i], p2_prime[i], p3_prime[i])

			v1[i + 1] = v1[i] + dv1 * delta_t
			v2[i + 1] = v2[i] + dv2 * delta_t
			v3[i + 1] = v3[i] + dv3 * delta_t

			p1[i + 1] = p1[i] + v1[i] * delta_t
			p2[i + 1] = p2[i] + v2[i] * delta_t
			p3[i + 1] = p3[i] + v3[i] * delta_t

			v1_prime[i + 1] = v1_prime[i] + dv1_prime * delta_t
			v2_prime[i + 1] = v2_prime[i] + dv2_prime * delta_t
			v3_prime[i + 1] = v3_prime[i] + dv3_prime * delta_t

			p1_prime[i + 1] = p1_prime[i] + v1_prime[i] * delta_t
			p2_prime[i + 1] = p2_prime[i] + v2_prime[i] * delta_t
			p3_prime[i + 1] = p3_prime[i] + v3_prime[i] * delta_t

			if i % 1000 == 0:
				fig = plt.figure(figsize=(10, 10))
				ax = fig.gca(projection='3d')
				plt.gca().patch.set_facecolor('black')
				ax.set_xlim([-50, 300])
				ax.set_ylim([-10, 30])
				ax.set_zlim([-30, 70])

				plt.plot([i[0] for i in p1], [j[1] for j in p1], [k[2] for k in p1] , '^', color='red', lw = 0.05, markersize = 0.01, alpha=0.5)
				plt.plot([i[0] for i in p2], [j[1] for j in p2], [k[2] for k in p2] , '^', color='white', lw = 0.05, markersize = 0.01, alpha=0.5)
				plt.plot([i[0] for i in p3], [j[1] for j in p3], [k[2] for k in p3] , '^', color='blue', lw = 0.05, markersize = 0.01, alpha=0.5)
				plt.plot([i[0] for i in p1_prime], [j[1] for j in p1_prime], [k[2] for k in p1_prime], '^', color='yellow', lw=0.05, markersize=0.01, alpha=0.5)

				plt.axis('on')

				# optional: use if reference axes skeleton is desired,
				# ie plt.axis is set to 'on'
				ax.set_xticks([]), ax.set_yticks([]), ax.set_zticks([])

				# make panes have the same color as background
				ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 1.0)), ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 1.0)), ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
				ax.view_init(elev = 20, azim = i//1000)
				plt.savefig('{}'.format(i//1000), bbox_inches='tight', dpi=300)
				plt.close()


		fig = plt.figure(figsize=(10, 10))
		ax = fig.gca(projection='3d')
		plt.gca().patch.set_facecolor('black')
		plt.plot([i[0] for i in p1], [j[1] for j in p1], [k[2] for k in p1] , '^', color='red', lw = 0.05, markersize = 0.01, alpha=0.5)
		plt.plot([i[0] for i in p1_prime], [j[1] for j in p1_prime], [k[2] for k in p1_prime] , '^', color='white', lw = 0.05, markersize = 0.01, alpha=0.5)
		plt.plot([i[0] for i in p1], [j[1] for j in p1], [k[2] for k in p1] , '^', color='red', lw = 0.05, markersize = 0.01, alpha=0.5)
		plt.plot([i[0] for i in p2], [j[1] for j in p2], [k[2] for k in p2] , '^', color='white', lw = 0.05, markersize = 0.01, alpha=0.5)
		plt.plot([i[0] for i in p3], [j[1] for j in p3], [k[2] for k in p3] , '^', color='blue', lw = 0.05, markersize = 0.01, alpha=0.5)


		plt.plot([i[0] for i in p1], [j[1] for j in p1], [k[2] for k in p1] , '^', color='red', lw = 0.05, markersize = 0.01, alpha=0.5)
		plt.plot([i[1] for i in p2], [np.sqrt(j[0]**2 + j[1]**2 + j[2]**2) for j in v2], ',', color='red', lw = 0.05, markersize = 0.01, alpha=0.8)
		plt.plot([i[0] for i in p3], [j[1] for j in p3], [k[2] for k in p3] , '^', color='blue', lw = 0.05, markersize = 0.01, alpha=0.5)
		plt.plot([i[0] for i in p2_prime], [j[1] for j in p2_prime], [k[2] for k in p2_prime], '^', color='blue', lw=0.05, markersize=0.01, alpha=0.5)
		plt.plot([i[1] for i in p2_prime], [np.sqrt(j[0]**2 + j[1]**2 + j[2]**2) for j in v2_prime], ',', color='blue', lw = 0.05, markersize = 0.01, alpha=0.8)

		plt.axis('on')

		# optional: use if reference axes skeleton is desired,
		# ie plt.axis is set to 'on'
		ax.set_xticks([]), ax.set_yticks([]), ax.set_zticks([])

		# make panes have the same color as background
		ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 1.0)), ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 1.0)), ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
		ax.view_init(elev = 20, azim = t)
		plt.savefig('{}'.format(t), dpi=300, bbox_inches='tight')
		plt.show()
		plt.close()


  
for i in range(1):
	time_steps = 50000
	x_res, y_res = 1000, 1000
	offset = -11
	mass = 30
	# print (f'Offset: {offset}')
	t = Threebody(time_steps, x_res, y_res, offset, mass)
	time_array = t.sensitivity(iterations_video=False, double_type=True)
	# t.three_body_trajectory()
	time_array = time_steps - time_array
	time_array = time_array.cpu().numpy()
	plt.style.use('dark_background')
	plt.imshow(time_array, cmap='inferno')
	plt.axis('off')
	plt.savefig('Threebody_divergence{0:04d}.png'.format(i+1), bbox_inches='tight', pad_inches=0, dpi=410)
	# plt.show()
	plt.close()
 

# time_steps = 50000
# x_res, y_res = 1000, 1000
# offset = -11
# mass = 30
# # print (f'Offset: {offset}')
# t = Threebody(time_steps, x_res, y_res, offset, mass)
# time_array = t.sensitivity(iterations_video=False, double_type=False)
# # t.three_body_trajectory()
# time_array_d = time_steps - time_array

# time_steps = 20000
# x_res, y_res = 300, 300
# offset = -11
# mass = 30
# # print (f'Offset: {offset}')
# t2 = Threebody(time_steps, x_res, y_res, offset, mass)
# time_array = t2.sensitivity(iterations_video=False, double_type=False)
# # t.three_body_trajectory()
# time_array_f = time_steps - time_array

# mask = time_array_d != time_array_f
# mask = mask.reshape([1, 300, 300])
# print (t2.p1[1:2][mask])
# print (t2.p1_prime[1:2][mask])


 
