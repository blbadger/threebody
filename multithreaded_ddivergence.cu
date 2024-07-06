#include <stdio.h>
#include <iostream>
#include <chrono>
#include <omp.h>
#include <thread>
#include <vector>

#include <cuda.h>
#include <cuda_runtime_api.h>

__global__
void saxpy(int n, float a, float *x)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  for (int j=0; j < 10000000; j++){
    if (i < n) {
      x[i] += 1;
    }
  }
}

// kernal declaration
__global__
void divergence(int n, 
              int steps,
              double delta_t,
              bool *still_together,
              bool *not_diverged,
              int *times,
              double m_1, double m_2, double m_3,
              double critical_distance,
              double *p1, double *p2, double *p3,
              double *p1_prime_x, double *p2_prime, double *p3_prime, 
              double *dv_1, double *dv_2, double *dv_3,
              double *dv_1pr, double *dv_2pr, double *dv_2pr,
              double *v1, double *v2, double *v3,
              double *v1_prime, double *v2_prime, double *v3_prime,
              double *nv1, double *nv2, double *nv3,
              double *nv1_prime, double *nv2_prime, double *nv3_prime
              )
{

  int i = blockIdx.x*blockDim.x + threadIdx.x;
  for (int j=0; j < steps; j++) {
    if (i < n and still_together[i]){
      // compute accelerations
      dv_1[i] = -9.8 * m_2 * (p1[i] - p2[i]) / pow(sqrt(pow(p1[i] - p2[i], 2) + pow(p1[i+1] - p2[i+1], 2) + pow(p1[i+2] - p2[i+2], 2)), 3) \
                  -9.8 * m_3 * (p1[i] - p3[i]) / pow(sqrt(pow(p1[i] - p3[i], 2) + pow(p1[i+1] - p3[i+1], 2) + pow(p1[i+2] - p3[i+2], 2)), 3);
      dv_1[i+1] = -9.8 * m_2 * (p1[i+1] - p2[i+1]) / pow(sqrt(pow(p1[i] - p2[i], 2) + pow(p1[i+1] - p2[i+1], 2) + pow(p1[i+2] - p2[i+2], 2)), 3) \
                  -9.8 * m_3 * (p1[i+1] - p3[i+1]) / pow(sqrt(pow(p1[i] - p3[i], 2) + pow(p1[i+1] - p3[i+1], 2) + pow(p1[i+2] - p3[i+2], 2)), 3);
      dv_1[i+2] = -9.8 * m_2 * (p1[i+2] - p2[i+2]) / pow(sqrt(pow(p1[i] - p2[i], 2) + pow(p1[i+1] - p2[i+1], 2) + pow(p1[i+2] - p2[i+2], 2)), 3) \
                  -9.8 * m_3 * (p1[i+2] - p3[i+2]) / pow(sqrt(pow(p1[i] - p3[i], 2) + pow(p1[i+1] - p3[i+1], 2) + pow(p1[i+2] - p3[i+2], 2)), 3);
      dv_2[i] = -9.8 * m_3 * (p2[i] - p3[i]) / pow(sqrt(pow(p2[i] - p3[i], 2) + pow(p2[i+1] - p3[i+1], 2) + pow(p2[i+2] - p3[i+2], 2)), 3) \
                  -9.8 * m_1 * (p2[i] - p1[i]) / pow(sqrt(pow(p2[i] - p1[i], 2) + pow(p2[i+1] - p1[i+1], 2) + pow(p2[i+2] - p1[i+2], 2)), 3);
      dv_2[i+1] = -9.8 * m_3 * (p2[i+1] - p3[i+1]) / pow(sqrt(pow(p2[i] - p3[i], 2) + pow(p2[i+1] - p3[i+1], 2) + pow(p2[i+2] - p3[i+2], 2)), 3) \
                  -9.8 * m_1 * (p2[i+1] - p1[i+1]) / pow(sqrt(pow(p2[i] - p1[i], 2) + pow(p2[i+1] - p1[i+1], 2) + pow(p2[i+2] - p1[i+2], 2)), 3);
      dv_2[i+2] = -9.8 * m_3 * (p2[i+2] - p3[i+2]) / pow(sqrt(pow(p2[i] - p3[i], 2) + pow(p2[i+1] - p3[i+1], 2) + pow(p2[i+2] - p3[i+2], 2)), 3) \
                  -9.8 * m_1 * (p2[i+2] - p1[i+2]) / pow(sqrt(pow(p2[i] - p1[i], 2) + pow(p2[i+1] - p1[i+1], 2) + pow(p2[i+2] - p1[i+2], 2)), 3);
      dv_3[i] = -9.8 * m_1 * (p3[i] - p1[i]) / pow(sqrt(pow(p3[i] - p1[i], 2) + pow(p3[i+1] - p1[i+1], 2) + pow(p3[i+2] - p1[i+2], 2)), 3) \
                  -9.8 * m_2 * (p3[i] - p2[i]) / pow(sqrt(pow(p3[i] - p2[i], 2) + pow(p3[i+1] - p2[i+1], 2) + pow(p3[i+2] - p2[i+2], 2)), 3);
      dv_3[i+1] = -9.8 * m_1 * (p3[i+1] - p1[i+1]) / pow(sqrt(pow(p3[i] - p1[i], 2) + pow(p3[i+1] - p1[i+1], 2) + pow(p3[i+2] - p1[i+2], 2)), 3) \
                  -9.8 * m_2 * (p3[i+1] - p2[i+1]) / pow(sqrt(pow(p3[i] - p2[i], 2) + pow(p3[i+1] - p2[i+1], 2) + pow(p3[i+2] - p2[i+2], 2)), 3);
      dv_3[i+2] = -9.8 * m_1 * (p3[i+2] - p1[i+2]) / pow(sqrt(pow(p3[i] - p1[i], 2) + pow(p3[i+1] - p1[i+1], 2) + pow(p3[i+2] - p1[i+2], 2)), 3) \
                  -9.8 * m_2 * (p3[i+2] - p2[i+2]) / pow(sqrt(pow(p3[i] - p2[i], 2) + pow(p3[i+1] - p2[i+1], 2) + pow(p3[i+2] - p2[i+2], 2)), 3);

      dv_1pr[i] = -9.8 * m_2 * (p1_prime[i] - p2_prime[i]) / pow(sqrt(pow(p1_prime[i] - p2_prime[i], 2) + pow(p1_prime[i+1] - p2_prime[i+1], 2) + pow(p1_prime[i+2] - p2_prime[i+2], 2)), 3) \
                    -9.8 * m_3 * (p1_prime[i] - p3_prime[i]) / pow(sqrt(pow(p1_prime[i] - p3_prime[i], 2) + pow(p1_prime[i+1] - p3_prime[i+1], 2) + pow(p1_prime[i+2] - p3_prime[i+2], 2)), 3);
      dv_1pr[i+1] = -9.8 * m_2 * (p1_prime[i+1] - p2_prime[i+1]) / pow(sqrt(pow(p1_prime[i] - p2_prime[i], 2) + pow(p1_prime[i+1] - p2_prime[i+1], 2) + pow(p1_prime[i+2] - p2_prime[i+2], 2)), 3) \
                    -9.8 * m_3 * (p1_prime[i+1] - p3_prime[i+1]) / pow(sqrt(pow(p1_prime[i] - p3_prime[i], 2) + pow(p1_prime[i+1] - p3_prime[i+1], 2) + pow(p1_prime[i+2] - p3_prime[i+2], 2)), 3);
      dv_1pr[i+2] = -9.8 * m_2 * (p1_prime[i+2] - p2_prime[i+2]) / pow(sqrt(pow(p1_prime[i] - p2_prime[i], 2) + pow(p1_prime[i+1] - p2_prime[i+1], 2) + pow(p1_prime[i+2] - p2_prime[i+2], 2)), 3) \
                    -9.8 * m_3 * (p1_prime[i+2] - p3_prime[i+2]) / pow(sqrt(pow(p1_prime[i] - p3_prime[i], 2) + pow(p1_prime[i+1] - p3_prime[i+1], 2) + pow(p1_prime[i+2] - p3_prime[i+2], 2)), 3);
      dv_2pr[i] = -9.8 * m_3 * (p2_prime[i] - p3_prime[i]) / pow(sqrt(pow(p2_prime[i] - p3_prime[i], 2) + pow(p2_prime[i+1] - p3_prime[i+1], 2) + pow(p2_prime[i+2] - p3_prime[i+2], 2)), 3) \
                    -9.8 * m_1 * (p2_prime[i] - p1_prime[i]) / pow(sqrt(pow(p2_prime[i] - p1_prime[i], 2) + pow(p2_prime[i+1] - p1_prime[i+1], 2) + pow(p2_prime[i+2] - p1_prime[i+2], 2)), 3);
      dv_2pr[i+1] = -9.8 * m_3 * (p2_prime[i+1] - p3_prime[i+1]) / pow(sqrt(pow(p2_prime[i] - p3_prime[i], 2) + pow(p2_prime[i+1] - p3_prime[i+1], 2) + pow(p2_prime[i+2] - p3_prime[i+2], 2)), 3) \
                    -9.8 * m_1 * (p2_prime[i+1] - p1_prime[i+1]) / pow(sqrt(pow(p2_prime[i] - p1_prime[i], 2) + pow(p2_prime[i+1] - p1_prime[i+1], 2) + pow(p2_prime[i+2] - p1_prime[i+2], 2)), 3);
      dv_2pr[i+2] = -9.8 * m_3 * (p2_prime[i+2] - p3_prime[i+2]) / pow(sqrt(pow(p2_prime[i] - p3_prime[i], 2) + pow(p2_prime[i+1] - p3_prime[i+1], 2) + pow(p2_prime[i+2] - p3_prime[i+2], 2)), 3) \
                    -9.8 * m_1 * (p2_prime[i+2] - p1_prime[i+2]) / pow(sqrt(pow(p2_prime[i] - p1_prime[i], 2) + pow(p2_prime[i+1] - p1_prime[i+1], 2) + pow(p2_prime[i+2] - p1_prime[i+2], 2)), 3);
      dv_3pr[i] = -9.8 * m_1 * (p3_prime[i] - p1_prime[i]) / pow(sqrt(pow(p3_prime[i] - p1_prime[i], 2) + pow(p3_prime[i+1] - p1_prime[i+1], 2) + pow(p3_prime[i+2] - p1_prime[i+2], 2)), 3) \
                    -9.8 * m_2 * (p3_prime[i] - p2_prime[i]) / pow(sqrt(pow(p3_prime[i] - p2_prime[i], 2) + pow(p3_prime[i+1] - p2_prime[i+1], 2) + pow(p3_prime[i+2] - p2_prime[i+2], 2)), 3);
      dv_3pr[i+1] = -9.8 * m_1 * (p3_prime[i+1] - p1_prime[i+1]) / pow(sqrt(pow(p3_prime[i] - p1_prime[i], 2) + pow(p3_prime[i+1] - p1_prime[i+1], 2) + pow(p3_prime[i+2] - p1_prime[i+2], 2)), 3) \
                    -9.8 * m_2 * (p3_prime[i+1] - p2_prime[i+1]) / pow(sqrt(pow(p3_prime[i] - p2_prime[i], 2) + pow(p3_prime[i+1] - p2_prime[i+1], 2) + pow(p3_prime[i+2] - p2_prime[i+2], 2)), 3);
      dv_3pr[i+2] = -9.8 * m_1 * (p3_prime[i+2] - p1_prime[i+2]) / pow(sqrt(pow(p3_prime[i] - p1_prime[i], 2) + pow(p3_prime[i+1] - p1_prime[i+1], 2) + pow(p3_prime[i+2] - p1_prime[i+2], 2)), 3) \
                    -9.8 * m_2 * (p3_prime[i+2] - p2_prime[i+2]) / pow(sqrt(pow(p3_prime[i] - p2_prime[i], 2) + pow(p3_prime[i+1] - p2_prime[i+1], 2) + pow(p3_prime[i+2] - p2_prime[i+2], 2)), 3);

      // find which trajectories have diverged and increment *times
      not_diverged[i] = (p1[i]-p1_prime[i])*(p1[i]-p1_prime[i]) + (p1[i+1]-p1_prime[i+1])*(p1[i+1]-p1_prime[i+1]) + (p1[i+2]-p1_prime[i+2])*(p1[i+2]-p1_prime[i+2]) <= critical_distance*critical_distance;
      still_together[i] = not_diverged[i] & still_together[i]; // bitwise and 
      if (still_together[i] == 1){
        times[i]++;
      };

      // compute new velocities
      nv1[i] = v1[i] + delta_t * dv_1[i];
      nv1[i+1] = v1[i+1] + delta_t * dv_1[i+1];
      nv1[i+2] = v1[i+2] + delta_t * dv_1[i+2];

      nv2[i] = v2[i] + delta_t * dv_2[i];
      nv2[i+1] = v2[i+1] + delta_t * dv_2[i+1];
      nv2[i+2] = v2[i+2] + delta_t * dv_2[i+2];

      nv3[i] = v3[i] + delta_t * dv_3[i];
      nv3[i+1] = v3[i+1] + delta_t * dv_3[i+1];
      nv3[i+2] = v3[i+2] + delta_t * dv_3[i+2];

      nv1_prime[i] = v1_prime[i] + delta_t * dv_1pr[i];
      nv1_prime[i+1] = v1_prime[i+1] + delta_t * dv_1pr[i+1];
      nv1_prime[i+2] = v1_prime[i+2] + delta_t * dv_1pr[i+2];

      nv2_prime[i] = v2_prime[i] + delta_t * dv_2pr[i];
      nv2_prime[i+1] = v2_prime[i+1] + delta_t * dv_2pr[i+1];
      nv2_prime[i+2] = v2_prime[i+2] + delta_t * dv_2pr[i+2];

      nv3_prime[i] = v3_prime[i] + delta_t * dv_3pr[i];
      nv3_prime[i+1] = v3_prime[i+1] + delta_t * dv_3pr[i+1];
      nv3_prime[i+2] = v3_prime[i+2] + delta_t * dv_3pr[i+2];

      // compute positions with current velocities
      p1[i] = p1[i] + delta_t * v1[i];
      p1[i+1] = p1[i+1] + delta_t * v1[i+1];
      p1[i+2] = p1[i+2] + delta_t * v1[i+2];

      p2[i] = p2[i] + delta_t * v2[i];
      p2[i+1] = p2[i+1] + delta_t * v2[i+1];
      p2[i+2] = p2[i+2] + delta_t * v2[i+2];

      p3[i] = p3[i] + delta_t * v3[i];
      p3[i+1] = p3[i+1] + delta_t * v3[i+1];
      p3[i+2] = p3[i+2] + delta_t * v3[i+2];

      p1_prime[i] = p1_prime[i] + delta_t * v1_prime[i];
      p1_prime[i+1] = p1_prime[i+1] + delta_t * v1_prime[i+1];
      p1_prime[i+2] = p1_prime[i+2] + delta_t * v1_prime[i+2];

      p2_prime[i] = p2_prime[i] + delta_t * v2_prime[i];
      p2_prime[i+1] = p2_prime[i+1] + delta_t * v2_prime[i+1];
      p2_prime[i+2] = p2_prime[i+2] + delta_t * v2_prime[i+2];

      p3_prime[i] = p3_prime[i] + delta_t * v3_prime[i];
      p3_prime[i+1] = p3_prime[i+1] + delta_t * v3_prime[i+1];
      p3_prime[i+2] = p3_prime[i+2] + delta_t * v3_prime[i+2];

      // assign new velocities to current velocities
      v1[i] = nv1[i];
      v1[i+1] = nv1[i+1];
      v1[i+2] = nv1[i+2];

      v2[i] = nv2[i];
      v2[i+1] = nv2[i+1];
      v2[i+2] = nv2[i+2];

      v3[i] = nv3[i];
      v3[i+1] = nv3[i+1];
      v3[i+2] = nv3[i+2];

      v1_prime[i] = nv1_prime[i];
      v1_prime[i+1] = nv1_prime[i+1];
      v1_prime[i+2] = nv1_prime[i+2];

      v2_prime[i] = nv2_prime[i];
      v2_prime[i+1] = nv2_prime[i+1];
      v2_prime[i+2] = nv2_prime[i+2];

      v3_prime[i] = nv3_prime[i];
      v3_prime[i+1] = nv3_prime[i+1];
      v3_prime[i+2] = nv3_prime[i+2];
      }
    }
  }

int main(void)
{
  // initialize timer
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  int N = 1000000;
  int steps = 50000;
  double delta_t = 0.001;
  double critical_distance = 0.5;
  double m1 = 10.;
  double m2 = 20.;
  double m3 = 30.;
  double *p1, *p2, *p3;
  double *p1_prime_x, *p1_prime_y, *p1_prime_z;
  double *dv_1, *dv_2, *dv_3;
  double *dv_1pr, *dv_2pr, *dv_3pr;
  double *v1_x, *v1_y, *v1_z;
  double *v1_prime, *v2_prime, *v3_prime;
  double *nv1, *nv2, *nv3;
  double *nv1_prime, *nv2_prime, *nv3_prime;

  double *d_p1, *d_p2, *d_p3;
  double *d_p1_prime, *d_p2_prime, *d_p3_prime;
  double *d_dv_1, *d_dv_2, *d_dv_3;
  double *d_dv_1pr, *d_dv_2pr, *d_dv_3pr;
  double *d_v1, *d_v2, *d_v3;
  double *d_v1_prime, *d_v2_prime, *d_v3_prime;
  double *d_nv1, *d_nv2, *d_nv3;
  double *d_nv1_prime, *d_nv2_prime, *d_nv3_prime;

  bool *still_together, *d_still_together;
  int *times, *d_times;
  bool *not_diverged, *d_not_diverged;

  cudaGetDeviceCount(&n_gpus);
  std::cout << n_gpus << " GPUs present. Allocating CPU memory and initializing values.\n";

  cudaHostAlloc((void**)&p1, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc((void**)&p2, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc((void**)&p3, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);

  cudaHostAlloc((void**)&p1_prime, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc((void**)&p2_prime, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc((void**)&p3_prime, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);

  cudaHostAlloc((void**)&dv_1, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc((void**)&dv_2, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc((void**)&dv_3, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);

  cudaHostAlloc((void**)&dv_1pr, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc((void**)&dv_2pr, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc((void**)&dv_3pr, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);

  cudaHostAlloc((void**)&v1, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc((void**)&v2, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc((void**)&v3, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);

  cudaHostAlloc((void**)&v1_prime, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);  
  cudaHostAlloc((void**)&v2_prime, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);  
  cudaHostAlloc((void**)&v3_prime, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);  

  cudaHostAlloc((void**)&nv1, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc((void**)&nv2, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc((void**)&nv3, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);

  cudaHostAlloc((void**)&nv1_prime, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc((void**)&nv2_prime, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc((void**)&nv3_prime, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);

  cudaHostAlloc((void**)&still_together, N*sizeof(bool), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc((void**)&not_diverged, N*sizeof(bool), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc((void**)&times, N*sizeof(int), cudaHostAllocWriteCombined | cudaHostAllocMapped);

  int resolution = sqrt(N);
  double range = 40;
  double step_size = range / resolution;
  for (int i = 0; i < N; i++) {
    int remainder = i % resolution;
    int step = i / resolution;
    p1[i] = -20. + 40*(double(remainder)/double(resolution));
    p1[i+N] = -20. + 40*(double(step)/double(resolution));
    p1[i+2*N] = -11.0;

    p2[i] = 0.0;
    p2[i+N] = 0.0;
    p2[i+2*N] = 0.0;

    p3[i] = 10.0;
    p3[i+N] = 10.0;
    p3[i+2*N] = 12.0;

    // shift p1 in all x, y, z vectors
    p1_prime[i] = -20. + 40*(double(remainder)/double(resolution)) + 0.001;
    p1_prime[i+N] = -20. + 40*(double(step)/double(resolution)) + 0.001;
    p1_prime[i+N*2] = -11.0 + 0.001;

    p2_prime[i] = 0.0;
    p2_prime[i+N] = 0.0;
    p2_prime[i+N*2] = 0.0;

    p3_prime_x[i] = 10.0;
    p3_prime_y[i+N] = 10.0;
    p3_prime_z[i+N*2] = 12.0;

    v1[i] = -3.;
    v1[i+N] = 0.;
    v1[i+N*2] = 0.;

    v2[i] = 0.;
    v2[i+N] = 0.;
    v2[i+N*2] = 0.;

    v3[i] = 3.;
    v3[i+N] = 0.;
    v3[i+N*2] = 0.;

    v1_prime[i] = -3.;
    v1_prime[i+N] = 0.;
    v1_prime[i+2*N] = 0.;

    v2_prime[i] = 0.;
    v2_prime[i+N] = 0.;
    v2_prime[i+N*2] = 0.;

    v3_prime[i] = 3.;
    v3_prime[i+N] = 0.;
    v3_prime[i+N*2] = 0.;

    times[i] = 0;
    still_together[i] = true;
    not_diverged[i] = true;
  }

  // launch one thread per GPU
  cudaStream_t streams[n_gpus];
  #pragma omp parallel num_threads(n_gpus)
  {
    int d = omp_get_thread_num();
    // assumes that n_gpus divides N with no remainder, which is safe as N is a large square.
    int start_idx = (N/n_gpus)*d;
    int end_idx = start_idx + N/n_gpus;
    std::cout << "Start index: " << start_idx << "\nEnd index: " << end_idx << "\n";
    int block_n = (N/n_gpus) * 3;
    cudaSetDevice(omp_get_thread_num());
    cudaStreamCreate(&streams[d]);

    cudaMalloc(&d_p1, block_n*sizeof(double)); 
    cudaMalloc(&d_p2, block_n*sizeof(double)); 
    cudaMalloc(&d_p3, block_n*sizeof(double));

    cudaMalloc(&d_p1_prime, block_n*sizeof(double));
    cudaMalloc(&d_p2_prime, block_n*sizeof(double));
    cudaMalloc(&d_p3_prime, block_n*sizeof(double));

    cudaMalloc(&d_dv_1, block_n*sizeof(double));
    cudaMalloc(&d_dv_2, block_n*sizeof(double));
    cudaMalloc(&d_dv_3, block_n*sizeof(double));

    cudaMalloc(&d_dv_1pr, block_n*sizeof(double));
    cudaMalloc(&d_dv_2pr, block_n*sizeof(double));
    cudaMalloc(&d_dv_3pr, block_n*sizeof(double));

    cudaMalloc(&d_v1, block_n*sizeof(double));
    cudaMalloc(&d_v2, block_n*sizeof(double));
    cudaMalloc(&d_v3, block_n*sizeof(double));

    cudaMalloc(&d_v1_prime, block_n*sizeof(double));
    cudaMalloc(&d_v2_prime, block_n*sizeof(double));
    cudaMalloc(&d_v3_prime, block_n*sizeof(double));

    cudaMalloc(&d_nv1, block_n*sizeof(double));
    cudaMalloc(&d_nv2, block_n*sizeof(double));
    cudaMalloc(&d_nv3, block_n*sizeof(double));

    cudaMalloc(&d_nv1_prime, block_n*sizeof(double));
    cudaMalloc(&d_nv2_prime, block_n*sizeof(double));
    cudaMalloc(&d_nv3_prime, block_n*sizeof(double));

    cudaMalloc(&d_still_together, block_n*sizeof(bool));
    cudaMalloc(&d_times, block_n*sizeof(int));
    cudaMalloc(&d_not_diverged, block_n*sizeof(bool));

    cudaMemcpyAsync(d_p1, p1+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_p2, p2+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_p3, p3+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);

    cudaMemcpyAsync(d_p1_prime, p1_prime_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_p2_prime, p2_prime_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_p3_prime, p3_prime_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);

    cudaMemcpyAsync(d_dv_1, dv_1_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_dv_2, dv_2_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_dv_3, dv_3_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);

    cudaMemcpyAsync(d_dv_1pr, dv_1pr_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_dv_2pr, dv_2pr_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_dv_3pr, dv_3pr_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);

    cudaMemcpyAsync(d_v1, v1_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_v2, v2_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_v3, v3_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);

    cudaMemcpyAsync(d_v1_prime, v1_prime_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_v2_prime, v2_prime_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_v3_prime, v3_prime_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);

    cudaMemcpyAsync(d_nv1, nv1_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_nv2, nv2_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_nv3, nv3_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);

    cudaMemcpyAsync(d_nv1_prime, nv1_prime_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_nv2_prime, nv1_prime_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_nv3_prime, nv1_prime_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);

    cudaMemcpyAsync(d_times, times+start_idx, block_n*sizeof(int), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_still_together, still_together+start_idx, block_n*sizeof(bool), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_not_diverged, not_diverged+start_idx, block_n*sizeof(bool), cudaMemcpyHostToDevice, streams[d]);

    // call CUDA kernal on inputs in configuration <<< blockIdx, threadIdx, 0, stream>>>>
    divergence<<<(N+255)/256, 256, 0, streams[d]>>>(
        block_n, 
        steps, 
        delta_t,
        d_still_together,
        d_not_diverged,
        d_times,
        m1, m2, m3,
        critical_distance,
        d_p1, 
        d_p2, 
        d_p3, 
        d_p1_prime, 
        d_p2_prime, 
        d_p3_prime,
        d_dv_1,
        d_dv_2,
        d_dv_3,
        d_dv_1pr,
        d_dv_2pr,
        d_dv_3pr,
        d_v1,
        d_v2,
        d_v3,
        d_v1_prime,
        d_v2_prime,
        d_v3_prime,
        d_nv1,
        d_nv2,
        d_nv3,
        d_nv1_prime,    
        d_nv2_prime,
        d_nv3_prime
        );

    cudaMemcpyAsync(times+start_idx, d_times, block_n*sizeof(int), cudaMemcpyDeviceToHost, streams[d]);
    cudaMemcpyAsync(still_together+start_idx, d_still_together, block_n*sizeof(bool), cudaMemcpyDeviceToHost, streams[d]);
    cudaMemcpyAsync(not_diverged+start_idx, d_not_diverged, block_n*sizeof(bool), cudaMemcpyDeviceToHost, streams[d]);
    cudaMemcpyAsync(p1+start_idx, d_p1_x, block_n*sizeof(double), cudaMemcpyDeviceToHost, streams[d]);
    cudaMemcpyAsync(p1_prime+start_idx, d_p1_prime_x, block_n*sizeof(double), cudaMemcpyDeviceToHost, streams[d]);
    cudaDeviceSynchronize();
    }

  // check computation for completion and accuracy
  for (int k=0; k<2; k++) {
    std::cout << times[k] << ' ';
    std::cout << still_together[k] << ' ';
    std::cout << not_diverged[k] << '\n';
    std::cout << p1_x[k] << ' ';
    std::cout << p1_y[k] << ' ';
    std::cout << p1_z[k] << ' ';
    std::cout << p1_prime_x[k] << ' ';
    std::cout << p1_prime_y[k] << ' ';
    std::cout << p1_prime_z[k] << ' ';
    std::cout << '\n';
 } 

  cudaError_t err = cudaGetLastError();  // add
  if (err != cudaSuccess) std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl; // add
  // cudaFree(d_p1_x); cudaFree(d_p1_y); cudaFree(d_p1_z);
  // cudaFree(d_p2_x); cudaFree(d_p2_y); cudaFree(d_p2_z);
  // cudaFree(d_p3_x); cudaFree(d_p3_y); cudaFree(d_p3_z);

  // cudaFree(d_p1_prime_x); cudaFree(d_p1_prime_y); cudaFree(d_p1_prime_z);  
  // cudaFree(d_p2_prime_x); cudaFree(d_p2_prime_y); cudaFree(d_p2_prime_z);
  // cudaFree(d_p3_prime_x); cudaFree(d_p3_prime_y); cudaFree(d_p3_prime_z);

  // cudaFree(d_dv_1_x); cudaFree(d_dv_1_y); cudaFree(d_dv_1_z);
  // cudaFree(d_dv_2_x); cudaFree(d_dv_2_y); cudaFree(d_dv_2_z);
  // cudaFree(d_dv_3_x); cudaFree(d_dv_3_y); cudaFree(d_dv_3_z);

  // cudaFree(d_dv_1pr_x); cudaFree(d_dv_1pr_y); cudaFree(d_dv_1pr_z);
  // cudaFree(d_dv_2pr_x); cudaFree(d_dv_2pr_y); cudaFree(d_dv_2pr_z);
  // cudaFree(d_dv_3pr_x); cudaFree(d_dv_3pr_y); cudaFree(d_dv_3pr_z);

  // cudaFree(d_nv1_x); cudaFree(d_nv1_y); cudaFree(d_nv1_z);
  // cudaFree(d_nv2_x); cudaFree(d_nv2_y); cudaFree(d_nv2_z);
  // cudaFree(d_nv3_x); cudaFree(d_nv3_y); cudaFree(d_nv3_z);

  // cudaFree(d_v1_x); cudaFree(d_v1_y); cudaFree(d_v1_z);
  // cudaFree(d_v2_x); cudaFree(d_v2_y); cudaFree(d_v2_z);
  // cudaFree(d_v3_x); cudaFree(d_v3_y); cudaFree(d_v3_z);

  // cudaFree(d_v1_prime_x); cudaFree(d_v1_prime_y); cudaFree(d_v1_prime_z);
  // cudaFree(d_v2_prime_x); cudaFree(d_v2_prime_y); cudaFree(d_v2_prime_z);
  // cudaFree(d_v3_prime_x); cudaFree(d_v3_prime_y); cudaFree(d_v3_prime_z);

  // cudaFree(d_nv1_prime_x); cudaFree(d_nv1_prime_y); cudaFree(d_nv1_prime_z);
  // cudaFree(d_nv2_prime_x); cudaFree(d_nv2_prime_y); cudaFree(d_nv2_prime_z);
  // cudaFree(d_nv3_prime_x); cudaFree(d_nv3_prime_y); cudaFree(d_nv3_prime_z);

  // cudaFree(d_times); cudaFree(d_still_together); cudaFree(d_not_diverged);

  // free(p1_x); free(p1_y); free(p1_z);
  // free(p2_x); free(p2_y); free(p2_z);
  // free(p3_x); free(p3_y); free(p3_z);

  // free(p1_prime_x); free(p1_prime_y); free(p1_prime_z);  
  // free(p2_prime_x); free(p2_prime_y); free(p2_prime_z);  
  // free(p3_prime_x); free(p3_prime_y); free(p3_prime_z);  

  // free(dv_1_x); free(dv_1_y); free(dv_1_z);
  // free(dv_2_x); free(dv_2_y); free(dv_2_z);
  // free(dv_3_x); free(dv_3_y); free(dv_3_z);

  // free(dv_1pr_x); free(dv_1pr_y); free(dv_1pr_z);
  // free(dv_2pr_x); free(dv_2pr_y); free(dv_2pr_z);
  // free(dv_3pr_x); free(dv_3pr_y); free(dv_3pr_z);

  // free(v1_x); free(v1_y); free(v1_z);
  // free(v2_x); free(v2_y); free(v2_z);
  // free(v3_x); free(v3_y); free(v3_z);

  // free(v1_prime_x); free(v1_prime_y); free(v1_prime_z);
  // free(v2_prime_x); free(v2_prime_y); free(v2_prime_z);
  // free(v3_prime_x); free(v3_prime_y); free(v3_prime_z);

  // free(nv1_x); free(nv1_y); free(nv1_z);
  // free(nv2_x); free(nv2_y); free(nv2_z);
  // free(nv3_x); free(nv3_y); free(nv3_z);

  // free(nv1_prime_x); free(nv1_prime_y); free(nv1_prime_z);
  // free(nv2_prime_x); free(nv2_prime_y); free(nv2_prime_z);
  // free(nv3_prime_x); free(nv3_prime_y); free(nv3_prime_z);
  // free(times); free(still_together); free(not_diverged);

  
  // measure elapsed kernal runtime
  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  std::cout << "Elapsed Time: " << elapsed_seconds.count() << "s\n";
  }

