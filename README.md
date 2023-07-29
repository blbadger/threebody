# 3-body-problem

Programs to simulate the trajectory of three astronomical bodies through space, moving according to Newton's gravitional laws.  Demonstrates periodicity 
in the general case for two but not three or more bodies.

`3_body_illustration.py` contains code to trace the path of three point objects in space.

![3body](https://blbadger.github.io/3_body_problem/3_body_shifted_1.png)

`divergence.py` contains modules to generate maps of how many iterations it takes for various starting points (of one body) to diverge if a small shift is made.  Code is now compatible with use of a GPU (if available) via `torch`.  Using a mid-level gaming GPU will speed up figure generation by a factor of around 60x.

![divergence](https://blbadger.github.io/3_body_problem/Threebody_divergence_xy.png)

`batched_divergence.py` implements a batched version of `divergence.py` in which multiple frames (ie different planet 1 mass divergence plots) are computed simultaneously. Depending on your GPU hardware (particularly the clock speed relative to the number of active CUDA cores at any given time) and the module resolution settings, this can lead to a modest speedup (~40%) or make very little difference. 
