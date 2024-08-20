# 3-body-problem

Programs to simulate the trajectory of three astronomical bodies through space, moving according to Newton's gravitional laws.  Demonstrates periodicity 
in the general case for two but not three or more bodies.

`3_body_illustration.py` contains code to trace the path of three point objects in space.

`divergence.py` contains modules to generate maps of how many iterations it takes for various starting points (of one body) to diverge if a small shift is made.  Code is now compatible with use of a GPU (if available) via `torch`.  Using a mid-level gaming GPU will speed up figure generation by a factor of around 60x.

`batched_divergence.py` implements a batched version of `divergence.py` in which multiple frames (ie different planet 1 mass divergence plots) are computed simultaneously. Depending on your GPU hardware (particularly the clock speed relative to the number of active CUDA cores at any given time) and the module resolution settings, this can lead to a modest speedup (~40%) or make very little difference. 

For further speedups via custom CUDA implementation, `divergence.cu` provides a highly optimized CUDA kernel with driver code to simulate three body divergence. `divergence_kernel.cu` and `divergence_transfer.py` implement a `ctypes`-based pointer transfer to drive an optimized CUDA kernal with python, followed by retrieval and Matplotlib-based plotting of the computed array. These kernels typically give a >2x speedup relative to the pytorch code as a result of these optimizations.

If you have multiple GPUs, `distributed_divergence.cu` provides a CUDA kernel and driver code that performs distributed integration, such that with N devices you can expect the total time taken to be 1/N times the time taken for one. With 4x V100 GPUs, you can expect to integrate a 1000x1000 grid of initial positions for 50,000 steps in around 18 seconds.

An single body's trajectory upon shifting relative to the other two bodies:

![3body](https://blbadger.github.io/3_body_problem/3_body_shifted_1.png)


A 1000x1000 (50k steps) divergence plot:

![divergence](https://blbadger.github.io/3_body_problem/Threebody_divergence_xy.png)


