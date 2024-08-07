# 3-body-problem

Programs to simulate the trajectory of three astronomical bodies through space, moving according to Newton's gravitional laws.  Demonstrates periodicity 
in the general case for two but not three or more bodies.

`3_body_illustration.py` contains code to trace the path of three point objects in space.

`divergence.py` contains modules to generate maps of how many iterations it takes for various starting points (of one body) to diverge if a small shift is made.  Code is now compatible with use of a GPU (if available) via `torch`.  Using a mid-level gaming GPU will speed up figure generation by a factor of around 60x.

`batched_divergence.py` implements a batched version of `divergence.py` in which multiple frames (ie different planet 1 mass divergence plots) are computed simultaneously. Depending on your GPU hardware (particularly the clock speed relative to the number of active CUDA cores at any given time) and the module resolution settings, this can lead to a modest speedup (~40%) or make very little difference. 

For further speedups via direct CUDA implementation, `divergence.cu` provides an optimized CUDA kernal with C++ driver code to simulate three body divergence. `divergence_kernal.cu` and `divergence_transfer.py` implement a `ctypes`-based pointer transfer to drive an optimized CUDA kernal with python, followed by retrieval and Matplotlib-based plotting of the computed array.

If you have multiple GPUs, `distributed_divergence.cu` provides a CUDA kernal and driver code that performs integration simultaneously on your devices, such that with N devices you can expect the time taken to be 1/N times the time taken for one. 

![3body](https://blbadger.github.io/3_body_problem/3_body_shifted_1.png)

![divergence](https://blbadger.github.io/3_body_problem/Threebody_divergence_xy.png)


