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


class Threebody:

	def __init__(self, time_steps):
		self.distance = 0.5
		self.m1 = 10
		self.m2 = 20
		self.m3 = 30
		self.time_steps = time_steps

		# assign a small time step
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
		planet_1_dv = -9.8 * m_2 * (p1 - p2)/(np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)**3) - \
					   9.8 * m_3 * (p1 - p3)/(np.sqrt((p1[0] - p3[0])**2 + (p1[1] - p3[1])**2 + (p1[2] - p3[2])**2)**3)

		planet_2_dv = -9.8 * m_3 * (p2 - p3)/(np.sqrt((p2[0] - p3[0])**2 + (p2[1] - p3[1])**2 + (p2[2] - p3[2])**2)**3) - \
					   9.8 * m_1 * (p2 - p1)/(np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 + (p2[2] - p1[2])**2)**3)

		planet_3_dv = -9.8 * m_1 * (p3 - p1)/(np.sqrt((p3[0] - p1[0])**2 + (p3[1] - p1[1])**2 + (p3[2] - p1[2])**2)**3) - \
					   9.8 * m_2 * (p3 - p2)/(np.sqrt((p3[0] - p2[0])**2 + (p3[1] - p2[1])**2 + (p3[2] - p2[2])**2)**3)

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
		separation_arr = np.sqrt((p1[0] - p1_prime[0])**2 + (p1[1] - p1_prime[1])**2 + (p1[2] - p1_prime[2])**2)
		bool_arr = separation_arr <= self.distance
		return bool_arr


	def sensitivity(self, y_res, x_res):
		"""
		Plots the sensitivity to initial values per starting point of planet 1, as
		measured by the time until divergence.

		Args:
			None

		Returns:
			None

		"""

		delta_t = self.delta_t
		y, z = np.arange(-20, 20, 40/y_res), np.arange(-30, 30, 60/x_res)
		grid = np.meshgrid(z, y)
		grid2 = np.meshgrid(z, y)

		# grid of all -11, identical starting z-values
		x = np.zeros(grid[0].shape) - 10

		# shift the grid by a small amount
		grid2 = grid2[0] + 1e-3, grid2[1] + 1e-3
		# grid of all -11, identical starting z-values
		x_prime = np.zeros(grid[0].shape) - 10 + 1e-3

		time_array = np.zeros(grid[0].shape)

		# starting coordinates for planets
		# p1_start = x_1, y_1, z_1
		p1 = np.array([x, grid[0], grid[1]])
		v1 = np.array([np.ones(grid[0].shape) * -3, np.zeros(grid[0].shape), np.zeros(grid[0].shape)])

		# p2_start = x_2, y_2, z_2
		p2 = np.array([np.zeros(grid[0].shape), np.zeros(grid[0].shape), np.zeros(grid[0].shape)])
		v2 = np.array([np.zeros(grid[0].shape), np.zeros(grid[0].shape), np.zeros(grid[0].shape)])

		# p3_start = x_3, y_3, z_3
		p3 = np.array([np.ones(grid[0].shape) * 10, np.ones(grid[0].shape) * 10, np.ones(grid[0].shape) * 12])
		v3 = np.array([np.ones(grid[0].shape) * 3, np.zeros(grid[0].shape), np.zeros(grid[0].shape)])

		# starting coordinates for planets shifted
		# p1_start = x_1, y_1, z_1
		p1_prime = np.array([x_prime, grid2[0], grid2[1]])
		v1_prime = np.array([np.ones(grid[0].shape) * -3, np.zeros(grid[0].shape), np.zeros(grid[0].shape)])

		# p2_start = x_2, y_2, z_2
		p2_prime = np.array([np.zeros(grid[0].shape), np.zeros(grid[0].shape), np.zeros(grid[0].shape)])
		v2_prime = np.array([np.zeros(grid[0].shape), np.zeros(grid[0].shape), np.zeros(grid[0].shape)])

		# p3_start = x_3, y_3, z_3
		p3_prime = np.array([np.ones(grid[0].shape) * 10, np.ones(grid[0].shape) * 10, np.ones(grid[0].shape) * 12])
		v3_prime = np.array([np.ones(grid[0].shape) * 3, np.zeros(grid[0].shape), np.zeros(grid[0].shape)])

		# bool array of all True
		still_together = grid[0] < 1e10
		t = time.time()
		p1, p2, p3 = torch.Tensor(p1), torch.Tensor(p2), torch.Tensor(p3)
		v1, v2, v3 = torch.Tensor(v1), torch.Tensor(v2), torch.Tensor(v3)
		p1, p2, p3 = p1.type(torch.double), p2.type(torch.double), p3.type(torch.double)
		v1, v2, v3 = v1.type(torch.double), v2.type(torch.double), v3.type(torch.double)

		p1_prime, p2_prime, p3_prime = torch.Tensor(p1_prime), torch.Tensor(p2_prime), torch.Tensor(p3_prime)
		v1_prime, v2_prime, v3_prime = torch.Tensor(v1_prime), torch.Tensor(v2_prime), torch.Tensor(v3_prime)

		# evolution of the system
		for i in range(self.time_steps):
			if i % 100 == 0:
				print (i)
				print (f'Elapsed time: {time.time() - t} seconds')
				time_array2 = i - time_array 
				plt.style.use('dark_background')
				plt.imshow(time_array2, cmap='inferno')
				plt.axis('off')
				plt.savefig('Threebody_divergence{0:04d}.png'.format(i//100), bbox_inches='tight', pad_inches=0, dpi=420)
				plt.close()

			not_diverged = self.not_diverged(p1, p1_prime)
			not_diverged = not_diverged.numpy()

			# points still together are not diverging now and have not previously
			still_together &= not_diverged

			# apply boolean mask to ndarray time_array
			time_array[still_together] += 1

			# calculate derivatives
			dv1, dv2, dv3 = self.accelerations(p1, p2, p3)
			dv1_prime, dv2_prime, dv3_prime = self.accelerations(p1_prime, p2_prime, p3_prime)

			nv1 = v1 + dv1 * delta_t
			nv2 = v2 + dv2 * delta_t
			nv3 = v3 + dv3 * delta_t

			p1 = p1 + v1 * delta_t
			p2 = p2 + v2 * delta_t
			p3 = p3 + v3 * delta_t
			v1, v2, v3 = nv1, nv2, nv3

			nv1_prime = v1_prime + dv1_prime * delta_t
			nv2_prime = v2_prime + dv2_prime * delta_t
			nv3_prime = v3_prime + dv3_prime * delta_t

			p1_prime = p1_prime + v1_prime * delta_t
			p2_prime = p2_prime + v2_prime * delta_t
			p3_prime = p3_prime + v3_prime * delta_t
			v1_prime, v2_prime, v3_prime = nv1_prime, nv2_prime, nv3_prime

		return time_array

	def three_body_phase(self):
		"""
		Plot the phase of three bodies according to Newtonian mechanics

		Args:
			None

		Returns:
			None

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
		p1_start_prime = np.array([-10.001, 10.001, -11.001])
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

			# if i % 1000 == 0:
			# 	fig = plt.figure(figsize=(10, 10))
			# 	ax = fig.gca(projection='3d')
			# 	plt.gca().patch.set_facecolor('black')
			# 	ax.set_xlim([-50, 300])
			# 	ax.set_ylim([-10, 30])
			# 	ax.set_zlim([-30, 70])

			# 	# plt.plot([i[0] for i in p1], [j[1] for j in p1], [k[2] for k in p1] , '^', color='red', lw = 0.05, markersize = 0.01, alpha=0.5)
			# 	plt.plot([i[0] for i in p2], [j[1] for j in p2], [k[2] for k in p2] , '^', color='white', lw = 0.05, markersize = 0.01, alpha=0.5)
			# 	plt.plot([i[0] for i in p3], [j[1] for j in p3], [k[2] for k in p3] , '^', color='blue', lw = 0.05, markersize = 0.01, alpha=0.5)
			# 	# plt.plot([i[0] for i in p1_prime], [j[1] for j in p1_prime], [k[2] for k in p1_prime], '^', color='blue', lw=0.05, markersize=0.01, alpha=0.5)

			# 	plt.axis('on')

			# 	# optional: use if reference axes skeleton is desired,
			# 	# ie plt.axis is set to 'on'
			# 	ax.set_xticks([]), ax.set_yticks([]), ax.set_zticks([])

			# 	# make panes have the same color as background
			# 	ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 1.0)), ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 1.0)), ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
			# 	ax.view_init(elev = 20, azim = i//1000)
			# 	plt.savefig('{}'.format(i//1000), bbox_inches='tight', dpi=300)
			# 	plt.close()


		# fig = plt.figure(figsize=(10, 10))
		# ax = fig.gca(projection='3d')
		# plt.gca().patch.set_facecolor('black')
		plt.plot([i[0] for i in p1], [j[1] for j in p1], [k[2] for k in p1] , '^', color='red', lw = 0.05, markersize = 0.01, alpha=0.5)
		plt.plot([i[0] for i in p1_prime], [j[1] for j in p1_prime], [k[2] for k in p1_prime] , '^', color='white', lw = 0.05, markersize = 0.01, alpha=0.5)
		# plt.plot([i[0] for i in p1], [j[1] for j in p1], [k[2] for k in p1] , '^', color='red', lw = 0.05, markersize = 0.01, alpha=0.5)
		# plt.plot([i[0] for i in p2], [j[1] for j in p2], [k[2] for k in p2] , '^', color='white', lw = 0.05, markersize = 0.01, alpha=0.5)
		# plt.plot([i[0] for i in p3], [j[1] for j in p3], [k[2] for k in p3] , '^', color='blue', lw = 0.05, markersize = 0.01, alpha=0.5)


		# plt.plot([i[0] for i in p1], [j[1] for j in p1], [k[2] for k in p1] , '^', color='red', lw = 0.05, markersize = 0.01, alpha=0.5)
		# plt.plot([i[1] for i in p2], [np.sqrt(j[0]**2 + j[1]**2 + j[2]**2) for j in v2], ',', color='red', lw = 0.05, markersize = 0.01, alpha=0.8)
		# plt.plot([i[0] for i in p3], [j[1] for j in p3], [k[2] for k in p3] , '^', color='blue', lw = 0.05, markersize = 0.01, alpha=0.5)
		# plt.plot([i[0] for i in p2_prime], [j[1] for j in p2_prime], [k[2] for k in p2_prime], '^', color='blue', lw=0.05, markersize=0.01, alpha=0.5)

		# plt.plot([i[1] for i in p2_prime], [np.sqrt(j[0]**2 + j[1]**2 + j[2]**2) for j in v2_prime], ',', color='blue', lw = 0.05, markersize = 0.01, alpha=0.8)

		plt.axis('on')

		# optional: use if reference axes skeleton is desired,
		# ie plt.axis is set to 'on'
		# ax.set_xticks([]), ax.set_yticks([]), ax.set_zticks([])

		# make panes have the same color as background
		# ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 1.0)), ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 1.0)), ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
		# ax.view_init(elev = 20, azim = t)
		# plt.savefig('{}'.format(t), dpi=300, bbox_inches='tight')
		plt.show()
		plt.close()

time_steps = 50000
t = Threebody(time_steps)
# t.three_body_trajectory()
time_array = t.sensitivity(1000, 1500)
# time_array = time_steps - time_array 
# plt.style.use('dark_background')
# plt.imshow(time_array, cmap='inferno')
# plt.axis('off')
# plt.savefig('Threebody_divergence{0:04d}.png'.format(3), bbox_inches='tight', pad_inches=0, dpi=410)
# plt.show()
# plt.close()












