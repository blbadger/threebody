

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
  for (int j=0; j < 100000000; j++){
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
              double *p1_x, double *p1_y, double *p1_z, 
              double *p2_x, double *p2_y, double *p2_z, 
              double *p3_x, double *p3_y, double *p3_z, 
              double *p1_prime_x, double *p1_prime_y, double *p1_prime_z, 
              double *p2_prime_x, double *p2_prime_y, double *p2_prime_z, 
              double *p3_prime_x, double *p3_prime_y, double *p3_prime_z, 
              double *dv_1_x, double *dv_1_y, double *dv_1_z,
              double *dv_2_x, double *dv_2_y, double *dv_2_z,
              double *dv_3_x, double *dv_3_y, double *dv_3_z,
              double *dv_1pr_x, double *dv_1pr_y, double *dv_1pr_z,
              double *dv_2pr_x, double *dv_2pr_y, double *dv_2pr_z,
              double *dv_3pr_x, double *dv_3pr_y, double *dv_3pr_z,
              double *v1_x, double *v1_y, double *v1_z,
              double *v2_x, double *v2_y, double *v2_z,
              double *v3_x, double *v3_y, double *v3_z,
              double *v1_prime_x, double *v1_prime_y, double *v1_prime_z,
              double *v2_prime_x, double *v2_prime_y, double *v2_prime_z,
              double *v3_prime_x, double *v3_prime_y, double *v3_prime_z,
              double *nv1_x, double *nv1_y, double *nv1_z,
              double *nv2_x, double *nv2_y, double *nv2_z,
              double *nv3_x, double *nv3_y, double *nv3_z,
              double *nv1_prime_x, double *nv1_prime_y, double *nv1_prime_z,
              double *nv2_prime_x, double *nv2_prime_y, double *nv2_prime_z,
              double *nv3_prime_x, double *nv3_prime_y, double *nv3_prime_z
              )
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  for (int j=0; j < steps; j++) {
    if (i < n and still_together[i]){
      // compute accelerations
      dv_1_x[i] = -9.8 * m_2 * (p1_x[i] - p2_x[i]) / (sqrt((p1_x[i] - p2_x[i])*(p1_x[i] - p2_x[i]) + (p1_y[i] - p2_y[i])*(p1_y[i] - p2_y[i]) + (p1_z[i] - p2_z[i])*(p1_z[i] - p2_z[i]))*sqrt((p1_x[i] - p2_x[i])*(p1_x[i] - p2_x[i]) + (p1_y[i] - p2_y[i])*(p1_y[i] - p2_y[i]) + (p1_z[i] - p2_z[i])*(p1_z[i] - p2_z[i]))*sqrt((p1_x[i] - p2_x[i])*(p1_x[i] - p2_x[i]) + (p1_y[i] - p2_y[i])*(p1_y[i] - p2_y[i]) + (p1_z[i] - p2_z[i])*(p1_z[i] - p2_z[i]))) -9.8 * m_3 * (p1_x[i] - p3_x[i]) / (sqrt((p1_x[i] - p3_x[i])*(p1_x[i] - p3_x[i]) + (p1_y[i] - p3_y[i])*(p1_y[i] - p3_y[i]) + (p1_z[i] - p3_z[i])*(p1_z[i] - p3_z[i]))*sqrt((p1_x[i] - p3_x[i])*(p1_x[i] - p3_x[i]) + (p1_y[i] - p3_y[i])*(p1_y[i] - p3_y[i]) + (p1_z[i] - p3_z[i])*(p1_z[i] - p3_z[i]))*sqrt((p1_x[i] - p3_x[i])*(p1_x[i] - p3_x[i]) + (p1_y[i] - p3_y[i])*(p1_y[i] - p3_y[i]) + (p1_z[i] - p3_z[i])*(p1_z[i] - p3_z[i])));
      dv_1_y[i] = -9.8 * m_2 * (p1_y[i] - p2_y[i]) / (sqrt((p1_x[i] - p2_x[i])*(p1_x[i] - p2_x[i]) + (p1_y[i] - p2_y[i])*(p1_y[i] - p2_y[i]) + (p1_z[i] - p2_z[i])*(p1_z[i] - p2_z[i]))*sqrt((p1_x[i] - p2_x[i])*(p1_x[i] - p2_x[i]) + (p1_y[i] - p2_y[i])*(p1_y[i] - p2_y[i]) + (p1_z[i] - p2_z[i])*(p1_z[i] - p2_z[i]))*sqrt((p1_x[i] - p2_x[i])*(p1_x[i] - p2_x[i]) + (p1_y[i] - p2_y[i])*(p1_y[i] - p2_y[i]) + (p1_z[i] - p2_z[i])*(p1_z[i] - p2_z[i]))) -9.8 * m_3 * (p1_y[i] - p3_y[i]) / (sqrt((p1_x[i] - p3_x[i])*(p1_x[i] - p3_x[i]) + (p1_y[i] - p3_y[i])*(p1_y[i] - p3_y[i]) + (p1_z[i] - p3_z[i])*(p1_z[i] - p3_z[i]))*sqrt((p1_x[i] - p3_x[i])*(p1_x[i] - p3_x[i]) + (p1_y[i] - p3_y[i])*(p1_y[i] - p3_y[i]) + (p1_z[i] - p3_z[i])*(p1_z[i] - p3_z[i]))*sqrt((p1_x[i] - p3_x[i])*(p1_x[i] - p3_x[i]) + (p1_y[i] - p3_y[i])*(p1_y[i] - p3_y[i]) + (p1_z[i] - p3_z[i])*(p1_z[i] - p3_z[i])));
      dv_1_z[i] = -9.8 * m_2 * (p1_z[i] - p2_z[i]) / (sqrt((p1_x[i] - p2_x[i])*(p1_x[i] - p2_x[i]) + (p1_y[i] - p2_y[i])*(p1_y[i] - p2_y[i]) + (p1_z[i] - p2_z[i])*(p1_z[i] - p2_z[i]))*sqrt((p1_x[i] - p2_x[i])*(p1_x[i] - p2_x[i]) + (p1_y[i] - p2_y[i])*(p1_y[i] - p2_y[i]) + (p1_z[i] - p2_z[i])*(p1_z[i] - p2_z[i]))*sqrt((p1_x[i] - p2_x[i])*(p1_x[i] - p2_x[i]) + (p1_y[i] - p2_y[i])*(p1_y[i] - p2_y[i]) + (p1_z[i] - p2_z[i])*(p1_z[i] - p2_z[i]))) -9.8 * m_3 * (p1_z[i] - p3_z[i]) / (sqrt((p1_x[i] - p3_x[i])*(p1_x[i] - p3_x[i]) + (p1_y[i] - p3_y[i])*(p1_y[i] - p3_y[i]) + (p1_z[i] - p3_z[i])*(p1_z[i] - p3_z[i]))*sqrt((p1_x[i] - p3_x[i])*(p1_x[i] - p3_x[i]) + (p1_y[i] - p3_y[i])*(p1_y[i] - p3_y[i]) + (p1_z[i] - p3_z[i])*(p1_z[i] - p3_z[i]))*sqrt((p1_x[i] - p3_x[i])*(p1_x[i] - p3_x[i]) + (p1_y[i] - p3_y[i])*(p1_y[i] - p3_y[i]) + (p1_z[i] - p3_z[i])*(p1_z[i] - p3_z[i])));
      
      dv_2_x[i] = -9.8 * m_3 * (p2_x[i] - p3_x[i]) / (sqrt((p2_x[i] - p3_x[i])*(p2_x[i] - p3_x[i]) + (p2_y[i] - p3_y[i])*(p2_y[i] - p3_y[i]) + (p2_z[i] - p3_z[i])*(p2_z[i] - p3_z[i]))*sqrt((p2_x[i] - p3_x[i])*(p2_x[i] - p3_x[i]) + (p2_y[i] - p3_y[i])*(p2_y[i] - p3_y[i]) + (p2_z[i] - p3_z[i])*(p2_z[i] - p3_z[i]))*sqrt((p2_x[i] - p3_x[i])*(p2_x[i] - p3_x[i]) + (p2_y[i] - p3_y[i])*(p2_y[i] - p3_y[i]) + (p2_z[i] - p3_z[i])*(p2_z[i] - p3_z[i]))) -9.8 * m_1 * (p2_x[i] - p1_x[i]) / (sqrt((p2_x[i] - p1_x[i])*(p2_x[i] - p1_x[i]) + (p2_y[i] - p1_y[i])*(p2_y[i] - p1_y[i]) + (p2_z[i] - p1_z[i])*(p2_z[i] - p1_z[i]))*sqrt((p2_x[i] - p1_x[i])*(p2_x[i] - p1_x[i]) + (p2_y[i] - p1_y[i])*(p2_y[i] - p1_y[i]) + (p2_z[i] - p1_z[i])*(p2_z[i] - p1_z[i]))*sqrt((p2_x[i] - p1_x[i])*(p2_x[i] - p1_x[i]) + (p2_y[i] - p1_y[i])*(p2_y[i] - p1_y[i]) + (p2_z[i] - p1_z[i])*(p2_z[i] - p1_z[i])));
      dv_2_y[i] = -9.8 * m_3 * (p2_y[i] - p3_y[i]) / (sqrt((p2_x[i] - p3_x[i])*(p2_x[i] - p3_x[i]) + (p2_y[i] - p3_y[i])*(p2_y[i] - p3_y[i]) + (p2_z[i] - p3_z[i])*(p2_z[i] - p3_z[i]))*sqrt((p2_x[i] - p3_x[i])*(p2_x[i] - p3_x[i]) + (p2_y[i] - p3_y[i])*(p2_y[i] - p3_y[i]) + (p2_z[i] - p3_z[i])*(p2_z[i] - p3_z[i]))*sqrt((p2_x[i] - p3_x[i])*(p2_x[i] - p3_x[i]) + (p2_y[i] - p3_y[i])*(p2_y[i] - p3_y[i]) + (p2_z[i] - p3_z[i])*(p2_z[i] - p3_z[i]))) -9.8 * m_1 * (p2_y[i] - p1_y[i]) / (sqrt((p2_x[i] - p1_x[i])*(p2_x[i] - p1_x[i]) + (p2_y[i] - p1_y[i])*(p2_y[i] - p1_y[i]) + (p2_z[i] - p1_z[i])*(p2_z[i] - p1_z[i]))*sqrt((p2_x[i] - p1_x[i])*(p2_x[i] - p1_x[i]) + (p2_y[i] - p1_y[i])*(p2_y[i] - p1_y[i]) + (p2_z[i] - p1_z[i])*(p2_z[i] - p1_z[i]))*sqrt((p2_x[i] - p1_x[i])*(p2_x[i] - p1_x[i]) + (p2_y[i] - p1_y[i])*(p2_y[i] - p1_y[i]) + (p2_z[i] - p1_z[i])*(p2_z[i] - p1_z[i])));
      dv_2_z[i] = -9.8 * m_3 * (p2_z[i] - p3_z[i]) / (sqrt((p2_x[i] - p3_x[i])*(p2_x[i] - p3_x[i]) + (p2_y[i] - p3_y[i])*(p2_y[i] - p3_y[i]) + (p2_z[i] - p3_z[i])*(p2_z[i] - p3_z[i]))*sqrt((p2_x[i] - p3_x[i])*(p2_x[i] - p3_x[i]) + (p2_y[i] - p3_y[i])*(p2_y[i] - p3_y[i]) + (p2_z[i] - p3_z[i])*(p2_z[i] - p3_z[i]))*sqrt((p2_x[i] - p3_x[i])*(p2_x[i] - p3_x[i]) + (p2_y[i] - p3_y[i])*(p2_y[i] - p3_y[i]) + (p2_z[i] - p3_z[i])*(p2_z[i] - p3_z[i]))) -9.8 * m_1 * (p2_z[i] - p1_z[i]) / (sqrt((p2_x[i] - p1_x[i])*(p2_x[i] - p1_x[i]) + (p2_y[i] - p1_y[i])*(p2_y[i] - p1_y[i]) + (p2_z[i] - p1_z[i])*(p2_z[i] - p1_z[i]))*sqrt((p2_x[i] - p1_x[i])*(p2_x[i] - p1_x[i]) + (p2_y[i] - p1_y[i])*(p2_y[i] - p1_y[i]) + (p2_z[i] - p1_z[i])*(p2_z[i] - p1_z[i]))*sqrt((p2_x[i] - p1_x[i])*(p2_x[i] - p1_x[i]) + (p2_y[i] - p1_y[i])*(p2_y[i] - p1_y[i]) + (p2_z[i] - p1_z[i])*(p2_z[i] - p1_z[i])));

      dv_3_x[i] = -9.8 * m_1 * (p3_x[i] - p1_x[i]) / (sqrt((p3_x[i] - p1_x[i])*(p3_x[i] - p1_x[i]) + (p3_y[i] - p1_y[i])*(p3_y[i] - p1_y[i]) + (p3_z[i] - p1_z[i])*(p3_z[i] - p1_z[i]))*sqrt((p3_x[i] - p1_x[i])*(p3_x[i] - p1_x[i]) + (p3_y[i] - p1_y[i])*(p3_y[i] - p1_y[i]) + (p3_z[i] - p1_z[i])*(p3_z[i] - p1_z[i]))*sqrt((p3_x[i] - p1_x[i])*(p3_x[i] - p1_x[i]) + (p3_y[i] - p1_y[i])*(p3_y[i] - p1_y[i]) + (p3_z[i] - p1_z[i])*(p3_z[i] - p1_z[i]))) -9.8 * m_2 * (p3_x[i] - p2_x[i]) / (sqrt((p3_x[i] - p2_x[i])*(p3_x[i] - p2_x[i]) + (p3_y[i] - p2_y[i])*(p3_y[i] - p2_y[i]) + (p3_z[i] - p2_z[i])*(p3_z[i] - p2_z[i]))*sqrt((p3_x[i] - p2_x[i])*(p3_x[i] - p2_x[i]) + (p3_y[i] - p2_y[i])*(p3_y[i] - p2_y[i]) + (p3_z[i] - p2_z[i])*(p3_z[i] - p2_z[i]))*sqrt((p3_x[i] - p2_x[i])*(p3_x[i] - p2_x[i]) + (p3_y[i] - p2_y[i])*(p3_y[i] - p2_y[i]) + (p3_z[i] - p2_z[i])*(p3_z[i] - p2_z[i])));
      dv_3_y[i] = -9.8 * m_1 * (p3_y[i] - p1_y[i]) / (sqrt((p3_x[i] - p1_x[i])*(p3_x[i] - p1_x[i]) + (p3_y[i] - p1_y[i])*(p3_y[i] - p1_y[i]) + (p3_z[i] - p1_z[i])*(p3_z[i] - p1_z[i]))*sqrt((p3_x[i] - p1_x[i])*(p3_x[i] - p1_x[i]) + (p3_y[i] - p1_y[i])*(p3_y[i] - p1_y[i]) + (p3_z[i] - p1_z[i])*(p3_z[i] - p1_z[i]))*sqrt((p3_x[i] - p1_x[i])*(p3_x[i] - p1_x[i]) + (p3_y[i] - p1_y[i])*(p3_y[i] - p1_y[i]) + (p3_z[i] - p1_z[i])*(p3_z[i] - p1_z[i]))) -9.8 * m_2 * (p3_y[i] - p2_y[i]) / (sqrt((p3_x[i] - p2_x[i])*(p3_x[i] - p2_x[i]) + (p3_y[i] - p2_y[i])*(p3_y[i] - p2_y[i]) + (p3_z[i] - p2_z[i])*(p3_z[i] - p2_z[i]))*sqrt((p3_x[i] - p2_x[i])*(p3_x[i] - p2_x[i]) + (p3_y[i] - p2_y[i])*(p3_y[i] - p2_y[i]) + (p3_z[i] - p2_z[i])*(p3_z[i] - p2_z[i]))*sqrt((p3_x[i] - p2_x[i])*(p3_x[i] - p2_x[i]) + (p3_y[i] - p2_y[i])*(p3_y[i] - p2_y[i]) + (p3_z[i] - p2_z[i])*(p3_z[i] - p2_z[i])));
      dv_3_z[i] = -9.8 * m_1 * (p3_z[i] - p1_z[i]) / (sqrt((p3_x[i] - p1_x[i])*(p3_x[i] - p1_x[i]) + (p3_y[i] - p1_y[i])*(p3_y[i] - p1_y[i]) + (p3_z[i] - p1_z[i])*(p3_z[i] - p1_z[i]))*sqrt((p3_x[i] - p1_x[i])*(p3_x[i] - p1_x[i]) + (p3_y[i] - p1_y[i])*(p3_y[i] - p1_y[i]) + (p3_z[i] - p1_z[i])*(p3_z[i] - p1_z[i]))*sqrt((p3_x[i] - p1_x[i])*(p3_x[i] - p1_x[i]) + (p3_y[i] - p1_y[i])*(p3_y[i] - p1_y[i]) + (p3_z[i] - p1_z[i])*(p3_z[i] - p1_z[i]))) -9.8 * m_2 * (p3_z[i] - p2_z[i]) / (sqrt((p3_x[i] - p2_x[i])*(p3_x[i] - p2_x[i]) + (p3_y[i] - p2_y[i])*(p3_y[i] - p2_y[i]) + (p3_z[i] - p2_z[i])*(p3_z[i] - p2_z[i]))*sqrt((p3_x[i] - p2_x[i])*(p3_x[i] - p2_x[i]) + (p3_y[i] - p2_y[i])*(p3_y[i] - p2_y[i]) + (p3_z[i] - p2_z[i])*(p3_z[i] - p2_z[i]))*sqrt((p3_x[i] - p2_x[i])*(p3_x[i] - p2_x[i]) + (p3_y[i] - p2_y[i])*(p3_y[i] - p2_y[i]) + (p3_z[i] - p2_z[i])*(p3_z[i] - p2_z[i])));

      dv_1pr_x[i] = -9.8 * m_2 * (p1_prime_x[i] - p2_prime_x[i]) / (sqrt((p1_prime_x[i] - p2_prime_x[i])*(p1_prime_x[i] - p2_prime_x[i]) + (p1_prime_y[i] - p2_prime_y[i])*(p1_prime_y[i] - p2_prime_y[i]) + (p1_prime_z[i] - p2_prime_z[i])*(p1_prime_z[i] - p2_prime_z[i]))*sqrt((p1_prime_x[i] - p2_prime_x[i])*(p1_prime_x[i] - p2_prime_x[i]) + (p1_prime_y[i] - p2_prime_y[i])*(p1_prime_y[i] - p2_prime_y[i]) + (p1_prime_z[i] - p2_prime_z[i])*(p1_prime_z[i] - p2_prime_z[i]))*sqrt((p1_prime_x[i] - p2_prime_x[i])*(p1_prime_x[i] - p2_prime_x[i]) + (p1_prime_y[i] - p2_prime_y[i])*(p1_prime_y[i] - p2_prime_y[i]) + (p1_prime_z[i] - p2_prime_z[i])*(p1_prime_z[i] - p2_prime_z[i]))) -9.8 * m_3 * (p1_prime_x[i] - p3_prime_x[i]) / (sqrt((p1_prime_x[i] - p3_prime_x[i])*(p1_prime_x[i] - p3_prime_x[i]) + (p1_prime_y[i] - p3_prime_y[i])*(p1_prime_y[i] - p3_prime_y[i]) + (p1_prime_z[i] - p3_prime_z[i])*(p1_prime_z[i] - p3_prime_z[i]))*sqrt((p1_prime_x[i] - p3_prime_x[i])*(p1_prime_x[i] - p3_prime_x[i]) + (p1_prime_y[i] - p3_prime_y[i])*(p1_prime_y[i] - p3_prime_y[i]) + (p1_prime_z[i] - p3_prime_z[i])*(p1_prime_z[i] - p3_prime_z[i]))*sqrt((p1_prime_x[i] - p3_prime_x[i])*(p1_prime_x[i] - p3_prime_x[i]) + (p1_prime_y[i] - p3_prime_y[i])*(p1_prime_y[i] - p3_prime_y[i]) + (p1_prime_z[i] - p3_prime_z[i])*(p1_prime_z[i] - p3_prime_z[i])));
      dv_1pr_y[i] = -9.8 * m_2 * (p1_prime_y[i] - p2_prime_y[i]) / (sqrt((p1_prime_x[i] - p2_prime_x[i])*(p1_prime_x[i] - p2_prime_x[i]) + (p1_prime_y[i] - p2_prime_y[i])*(p1_prime_y[i] - p2_prime_y[i]) + (p1_prime_z[i] - p2_prime_z[i])*(p1_prime_z[i] - p2_prime_z[i]))*sqrt((p1_prime_x[i] - p2_prime_x[i])*(p1_prime_x[i] - p2_prime_x[i]) + (p1_prime_y[i] - p2_prime_y[i])*(p1_prime_y[i] - p2_prime_y[i]) + (p1_prime_z[i] - p2_prime_z[i])*(p1_prime_z[i] - p2_prime_z[i]))*sqrt((p1_prime_x[i] - p2_prime_x[i])*(p1_prime_x[i] - p2_prime_x[i]) + (p1_prime_y[i] - p2_prime_y[i])*(p1_prime_y[i] - p2_prime_y[i]) + (p1_prime_z[i] - p2_prime_z[i])*(p1_prime_z[i] - p2_prime_z[i]))) -9.8 * m_3 * (p1_prime_y[i] - p3_prime_y[i]) / (sqrt((p1_prime_x[i] - p3_prime_x[i])*(p1_prime_x[i] - p3_prime_x[i]) + (p1_prime_y[i] - p3_prime_y[i])*(p1_prime_y[i] - p3_prime_y[i]) + (p1_prime_z[i] - p3_prime_z[i])*(p1_prime_z[i] - p3_prime_z[i]))*sqrt((p1_prime_x[i] - p3_prime_x[i])*(p1_prime_x[i] - p3_prime_x[i]) + (p1_prime_y[i] - p3_prime_y[i])*(p1_prime_y[i] - p3_prime_y[i]) + (p1_prime_z[i] - p3_prime_z[i])*(p1_prime_z[i] - p3_prime_z[i]))*sqrt((p1_prime_x[i] - p3_prime_x[i])*(p1_prime_x[i] - p3_prime_x[i]) + (p1_prime_y[i] - p3_prime_y[i])*(p1_prime_y[i] - p3_prime_y[i]) + (p1_prime_z[i] - p3_prime_z[i])*(p1_prime_z[i] - p3_prime_z[i])));
      dv_1pr_z[i] = -9.8 * m_2 * (p1_prime_z[i] - p2_prime_z[i]) / (sqrt((p1_prime_x[i] - p2_prime_x[i])*(p1_prime_x[i] - p2_prime_x[i]) + (p1_prime_y[i] - p2_prime_y[i])*(p1_prime_y[i] - p2_prime_y[i]) + (p1_prime_z[i] - p2_prime_z[i])*(p1_prime_z[i] - p2_prime_z[i]))*sqrt((p1_prime_x[i] - p2_prime_x[i])*(p1_prime_x[i] - p2_prime_x[i]) + (p1_prime_y[i] - p2_prime_y[i])*(p1_prime_y[i] - p2_prime_y[i]) + (p1_prime_z[i] - p2_prime_z[i])*(p1_prime_z[i] - p2_prime_z[i]))*sqrt((p1_prime_x[i] - p2_prime_x[i])*(p1_prime_x[i] - p2_prime_x[i]) + (p1_prime_y[i] - p2_prime_y[i])*(p1_prime_y[i] - p2_prime_y[i]) + (p1_prime_z[i] - p2_prime_z[i])*(p1_prime_z[i] - p2_prime_z[i]))) -9.8 * m_3 * (p1_prime_z[i] - p3_prime_z[i]) / (sqrt((p1_prime_x[i] - p3_prime_x[i])*(p1_prime_x[i] - p3_prime_x[i]) + (p1_prime_y[i] - p3_prime_y[i])*(p1_prime_y[i] - p3_prime_y[i]) + (p1_prime_z[i] - p3_prime_z[i])*(p1_prime_z[i] - p3_prime_z[i]))*sqrt((p1_prime_x[i] - p3_prime_x[i])*(p1_prime_x[i] - p3_prime_x[i]) + (p1_prime_y[i] - p3_prime_y[i])*(p1_prime_y[i] - p3_prime_y[i]) + (p1_prime_z[i] - p3_prime_z[i])*(p1_prime_z[i] - p3_prime_z[i]))*sqrt((p1_prime_x[i] - p3_prime_x[i])*(p1_prime_x[i] - p3_prime_x[i]) + (p1_prime_y[i] - p3_prime_y[i])*(p1_prime_y[i] - p3_prime_y[i]) + (p1_prime_z[i] - p3_prime_z[i])*(p1_prime_z[i] - p3_prime_z[i])));
      
      dv_2pr_x[i] = -9.8 * m_3 * (p2_prime_x[i] - p3_prime_x[i]) / (sqrt((p2_prime_x[i] - p3_prime_x[i])*(p2_prime_x[i] - p3_prime_x[i]) + (p2_prime_y[i] - p3_prime_y[i])*(p2_prime_y[i] - p3_prime_y[i]) + (p2_prime_z[i] - p3_prime_z[i])*(p2_prime_z[i] - p3_prime_z[i]))*sqrt((p2_prime_x[i] - p3_prime_x[i])*(p2_prime_x[i] - p3_prime_x[i]) + (p2_prime_y[i] - p3_prime_y[i])*(p2_prime_y[i] - p3_prime_y[i]) + (p2_prime_z[i] - p3_prime_z[i])*(p2_prime_z[i] - p3_prime_z[i]))*sqrt((p2_prime_x[i] - p3_prime_x[i])*(p2_prime_x[i] - p3_prime_x[i]) + (p2_prime_y[i] - p3_prime_y[i])*(p2_prime_y[i] - p3_prime_y[i]) + (p2_prime_z[i] - p3_prime_z[i])*(p2_prime_z[i] - p3_prime_z[i]))) -9.8 * m_1 * (p2_prime_x[i] - p1_prime_x[i]) / (sqrt((p2_prime_x[i] - p1_prime_x[i])*(p2_prime_x[i] - p1_prime_x[i]) + (p2_prime_y[i] - p1_prime_y[i])*(p2_prime_y[i] - p1_prime_y[i]) + (p2_prime_z[i] - p1_prime_z[i])*(p2_prime_z[i] - p1_prime_z[i]))*sqrt((p2_prime_x[i] - p1_prime_x[i])*(p2_prime_x[i] - p1_prime_x[i]) + (p2_prime_y[i] - p1_prime_y[i])*(p2_prime_y[i] - p1_prime_y[i]) + (p2_prime_z[i] - p1_prime_z[i])*(p2_prime_z[i] - p1_prime_z[i]))*sqrt((p2_prime_x[i] - p1_prime_x[i])*(p2_prime_x[i] - p1_prime_x[i]) + (p2_prime_y[i] - p1_prime_y[i])*(p2_prime_y[i] - p1_prime_y[i]) + (p2_prime_z[i] - p1_prime_z[i])*(p2_prime_z[i] - p1_prime_z[i])));
      dv_2pr_y[i] = -9.8 * m_3 * (p2_prime_y[i] - p3_prime_y[i]) / (sqrt((p2_prime_x[i] - p3_prime_x[i])*(p2_prime_x[i] - p3_prime_x[i]) + (p2_prime_y[i] - p3_prime_y[i])*(p2_prime_y[i] - p3_prime_y[i]) + (p2_prime_z[i] - p3_prime_z[i])*(p2_prime_z[i] - p3_prime_z[i]))*sqrt((p2_prime_x[i] - p3_prime_x[i])*(p2_prime_x[i] - p3_prime_x[i]) + (p2_prime_y[i] - p3_prime_y[i])*(p2_prime_y[i] - p3_prime_y[i]) + (p2_prime_z[i] - p3_prime_z[i])*(p2_prime_z[i] - p3_prime_z[i]))*sqrt((p2_prime_x[i] - p3_prime_x[i])*(p2_prime_x[i] - p3_prime_x[i]) + (p2_prime_y[i] - p3_prime_y[i])*(p2_prime_y[i] - p3_prime_y[i]) + (p2_prime_z[i] - p3_prime_z[i])*(p2_prime_z[i] - p3_prime_z[i]))) -9.8 * m_1 * (p2_prime_y[i] - p1_prime_y[i]) / (sqrt((p2_prime_x[i] - p1_prime_x[i])*(p2_prime_x[i] - p1_prime_x[i]) + (p2_prime_y[i] - p1_prime_y[i])*(p2_prime_y[i] - p1_prime_y[i]) + (p2_prime_z[i] - p1_prime_z[i])*(p2_prime_z[i] - p1_prime_z[i]))*sqrt((p2_prime_x[i] - p1_prime_x[i])*(p2_prime_x[i] - p1_prime_x[i]) + (p2_prime_y[i] - p1_prime_y[i])*(p2_prime_y[i] - p1_prime_y[i]) + (p2_prime_z[i] - p1_prime_z[i])*(p2_prime_z[i] - p1_prime_z[i]))*sqrt((p2_prime_x[i] - p1_prime_x[i])*(p2_prime_x[i] - p1_prime_x[i]) + (p2_prime_y[i] - p1_prime_y[i])*(p2_prime_y[i] - p1_prime_y[i]) + (p2_prime_z[i] - p1_prime_z[i])*(p2_prime_z[i] - p1_prime_z[i])));
      dv_2pr_z[i] = -9.8 * m_3 * (p2_prime_z[i] - p3_prime_z[i]) / (sqrt((p2_prime_x[i] - p3_prime_x[i])*(p2_prime_x[i] - p3_prime_x[i]) + (p2_prime_y[i] - p3_prime_y[i])*(p2_prime_y[i] - p3_prime_y[i]) + (p2_prime_z[i] - p3_prime_z[i])*(p2_prime_z[i] - p3_prime_z[i]))*sqrt((p2_prime_x[i] - p3_prime_x[i])*(p2_prime_x[i] - p3_prime_x[i]) + (p2_prime_y[i] - p3_prime_y[i])*(p2_prime_y[i] - p3_prime_y[i]) + (p2_prime_z[i] - p3_prime_z[i])*(p2_prime_z[i] - p3_prime_z[i]))*sqrt((p2_prime_x[i] - p3_prime_x[i])*(p2_prime_x[i] - p3_prime_x[i]) + (p2_prime_y[i] - p3_prime_y[i])*(p2_prime_y[i] - p3_prime_y[i]) + (p2_prime_z[i] - p3_prime_z[i])*(p2_prime_z[i] - p3_prime_z[i]))) -9.8 * m_1 * (p2_prime_z[i] - p1_prime_z[i]) / (sqrt((p2_prime_x[i] - p1_prime_x[i])*(p2_prime_x[i] - p1_prime_x[i]) + (p2_prime_y[i] - p1_prime_y[i])*(p2_prime_y[i] - p1_prime_y[i]) + (p2_prime_z[i] - p1_prime_z[i])*(p2_prime_z[i] - p1_prime_z[i]))*sqrt((p2_prime_x[i] - p1_prime_x[i])*(p2_prime_x[i] - p1_prime_x[i]) + (p2_prime_y[i] - p1_prime_y[i])*(p2_prime_y[i] - p1_prime_y[i]) + (p2_prime_z[i] - p1_prime_z[i])*(p2_prime_z[i] - p1_prime_z[i]))*sqrt((p2_prime_x[i] - p1_prime_x[i])*(p2_prime_x[i] - p1_prime_x[i]) + (p2_prime_y[i] - p1_prime_y[i])*(p2_prime_y[i] - p1_prime_y[i]) + (p2_prime_z[i] - p1_prime_z[i])*(p2_prime_z[i] - p1_prime_z[i])));

      dv_3pr_x[i] = -9.8 * m_1 * (p3_prime_x[i] - p1_prime_x[i]) / (sqrt((p3_prime_x[i] - p1_prime_x[i])*(p3_prime_x[i] - p1_prime_x[i]) + (p3_prime_y[i] - p1_prime_y[i])*(p3_prime_y[i] - p1_prime_y[i]) + (p3_prime_z[i] - p1_prime_z[i])*(p3_prime_z[i] - p1_prime_z[i]))*sqrt((p3_prime_x[i] - p1_prime_x[i])*(p3_prime_x[i] - p1_prime_x[i]) + (p3_prime_y[i] - p1_prime_y[i])*(p3_prime_y[i] - p1_prime_y[i]) + (p3_prime_z[i] - p1_prime_z[i])*(p3_prime_z[i] - p1_prime_z[i]))*sqrt((p3_prime_x[i] - p1_prime_x[i])*(p3_prime_x[i] - p1_prime_x[i]) + (p3_prime_y[i] - p1_prime_y[i])*(p3_prime_y[i] - p1_prime_y[i]) + (p3_prime_z[i] - p1_prime_z[i])*(p3_prime_z[i] - p1_prime_z[i]))) -9.8 * m_2 * (p3_prime_x[i] - p2_prime_x[i]) / (sqrt((p3_prime_x[i] - p2_prime_x[i])*(p3_prime_x[i] - p2_prime_x[i]) + (p3_prime_y[i] - p2_prime_y[i])*(p3_prime_y[i] - p2_prime_y[i]) + (p3_prime_z[i] - p2_prime_z[i])*(p3_prime_z[i] - p2_prime_z[i]))*sqrt((p3_prime_x[i] - p2_prime_x[i])*(p3_prime_x[i] - p2_prime_x[i]) + (p3_prime_y[i] - p2_prime_y[i])*(p3_prime_y[i] - p2_prime_y[i]) + (p3_prime_z[i] - p2_prime_z[i])*(p3_prime_z[i] - p2_prime_z[i]))*sqrt((p3_prime_x[i] - p2_prime_x[i])*(p3_prime_x[i] - p2_prime_x[i]) + (p3_prime_y[i] - p2_prime_y[i])*(p3_prime_y[i] - p2_prime_y[i]) + (p3_prime_z[i] - p2_prime_z[i])*(p3_prime_z[i] - p2_prime_z[i])));
      dv_3pr_y[i] = -9.8 * m_1 * (p3_prime_y[i] - p1_prime_y[i]) / (sqrt((p3_prime_x[i] - p1_prime_x[i])*(p3_prime_x[i] - p1_prime_x[i]) + (p3_prime_y[i] - p1_prime_y[i])*(p3_prime_y[i] - p1_prime_y[i]) + (p3_prime_z[i] - p1_prime_z[i])*(p3_prime_z[i] - p1_prime_z[i]))*sqrt((p3_prime_x[i] - p1_prime_x[i])*(p3_prime_x[i] - p1_prime_x[i]) + (p3_prime_y[i] - p1_prime_y[i])*(p3_prime_y[i] - p1_prime_y[i]) + (p3_prime_z[i] - p1_prime_z[i])*(p3_prime_z[i] - p1_prime_z[i]))*sqrt((p3_prime_x[i] - p1_prime_x[i])*(p3_prime_x[i] - p1_prime_x[i]) + (p3_prime_y[i] - p1_prime_y[i])*(p3_prime_y[i] - p1_prime_y[i]) + (p3_prime_z[i] - p1_prime_z[i])*(p3_prime_z[i] - p1_prime_z[i]))) -9.8 * m_2 * (p3_prime_y[i] - p2_prime_y[i]) / (sqrt((p3_prime_x[i] - p2_prime_x[i])*(p3_prime_x[i] - p2_prime_x[i]) + (p3_prime_y[i] - p2_prime_y[i])*(p3_prime_y[i] - p2_prime_y[i]) + (p3_prime_z[i] - p2_prime_z[i])*(p3_prime_z[i] - p2_prime_z[i]))*sqrt((p3_prime_x[i] - p2_prime_x[i])*(p3_prime_x[i] - p2_prime_x[i]) + (p3_prime_y[i] - p2_prime_y[i])*(p3_prime_y[i] - p2_prime_y[i]) + (p3_prime_z[i] - p2_prime_z[i])*(p3_prime_z[i] - p2_prime_z[i]))*sqrt((p3_prime_x[i] - p2_prime_x[i])*(p3_prime_x[i] - p2_prime_x[i]) + (p3_prime_y[i] - p2_prime_y[i])*(p3_prime_y[i] - p2_prime_y[i]) + (p3_prime_z[i] - p2_prime_z[i])*(p3_prime_z[i] - p2_prime_z[i])));
      dv_3pr_z[i] = -9.8 * m_1 * (p3_prime_z[i] - p1_prime_z[i]) / (sqrt((p3_prime_x[i] - p1_prime_x[i])*(p3_prime_x[i] - p1_prime_x[i]) + (p3_prime_y[i] - p1_prime_y[i])*(p3_prime_y[i] - p1_prime_y[i]) + (p3_prime_z[i] - p1_prime_z[i])*(p3_prime_z[i] - p1_prime_z[i]))*sqrt((p3_prime_x[i] - p1_prime_x[i])*(p3_prime_x[i] - p1_prime_x[i]) + (p3_prime_y[i] - p1_prime_y[i])*(p3_prime_y[i] - p1_prime_y[i]) + (p3_prime_z[i] - p1_prime_z[i])*(p3_prime_z[i] - p1_prime_z[i]))*sqrt((p3_prime_x[i] - p1_prime_x[i])*(p3_prime_x[i] - p1_prime_x[i]) + (p3_prime_y[i] - p1_prime_y[i])*(p3_prime_y[i] - p1_prime_y[i]) + (p3_prime_z[i] - p1_prime_z[i])*(p3_prime_z[i] - p1_prime_z[i]))) -9.8 * m_2 * (p3_prime_z[i] - p2_prime_z[i]) / (sqrt((p3_prime_x[i] - p2_prime_x[i])*(p3_prime_x[i] - p2_prime_x[i]) + (p3_prime_y[i] - p2_prime_y[i])*(p3_prime_y[i] - p2_prime_y[i]) + (p3_prime_z[i] - p2_prime_z[i])*(p3_prime_z[i] - p2_prime_z[i]))*sqrt((p3_prime_x[i] - p2_prime_x[i])*(p3_prime_x[i] - p2_prime_x[i]) + (p3_prime_y[i] - p2_prime_y[i])*(p3_prime_y[i] - p2_prime_y[i]) + (p3_prime_z[i] - p2_prime_z[i])*(p3_prime_z[i] - p2_prime_z[i]))*sqrt((p3_prime_x[i] - p2_prime_x[i])*(p3_prime_x[i] - p2_prime_x[i]) + (p3_prime_y[i] - p2_prime_y[i])*(p3_prime_y[i] - p2_prime_y[i]) + (p3_prime_z[i] - p2_prime_z[i])*(p3_prime_z[i] - p2_prime_z[i])));


      // find which trajectories have diverged and increment *times
      not_diverged[i] = (p1_x[i]-p1_prime_x[i])*(p1_x[i]-p1_prime_x[i]) + (p1_y[i]-p1_prime_y[i])*(p1_y[i]-p1_prime_y[i]) + (p1_z[i]-p1_prime_z[i])*(p1_z[i]-p1_prime_z[i]) <= critical_distance*critical_distance;
      still_together[i] = not_diverged[i] & still_together[i]; // bitwise and 
      if (still_together[i] == 1){
        times[i]++;
      };

      // compute new velocities
      nv1_x[i] = v1_x[i] + delta_t * dv_1_x[i];
      nv1_y[i] = v1_y[i] + delta_t * dv_1_y[i];
      nv1_z[i] = v1_z[i] + delta_t * dv_1_z[i];

      nv2_x[i] = v2_x[i] + delta_t * dv_2_x[i];
      nv2_y[i] = v2_y[i] + delta_t * dv_2_y[i];
      nv2_z[i] = v2_z[i] + delta_t * dv_2_z[i];

      nv3_x[i] = v3_x[i] + delta_t * dv_3_x[i];
      nv3_y[i] = v3_y[i] + delta_t * dv_3_y[i];
      nv3_z[i] = v3_z[i] + delta_t * dv_3_z[i];

      nv1_prime_x[i] = v1_prime_x[i] + delta_t * dv_1pr_x[i];
      nv1_prime_y[i] = v1_prime_y[i] + delta_t * dv_1pr_y[i];
      nv1_prime_z[i] = v1_prime_z[i] + delta_t * dv_1pr_z[i];

      nv2_prime_x[i] = v2_prime_x[i] + delta_t * dv_2pr_x[i];
      nv2_prime_y[i] = v2_prime_y[i] + delta_t * dv_2pr_y[i];
      nv2_prime_z[i] = v2_prime_z[i] + delta_t * dv_2pr_z[i];

      nv3_prime_x[i] = v3_prime_x[i] + delta_t * dv_3pr_x[i];
      nv3_prime_y[i] = v3_prime_y[i] + delta_t * dv_3pr_y[i];
      nv3_prime_z[i] = v3_prime_z[i] + delta_t * dv_3pr_z[i];

      // compute positions with current velocities
      p1_x[i] = p1_x[i] + delta_t * v1_x[i];
      p1_y[i] = p1_y[i] + delta_t * v1_y[i];
      p1_z[i] = p1_z[i] + delta_t * v1_z[i];

      p2_x[i] = p2_x[i] + delta_t * v2_x[i];
      p2_y[i] = p2_y[i] + delta_t * v2_y[i];
      p2_z[i] = p2_z[i] + delta_t * v2_z[i];

      p3_x[i] = p3_x[i] + delta_t * v3_x[i];
      p3_y[i] = p3_y[i] + delta_t * v3_y[i];
      p3_z[i] = p3_z[i] + delta_t * v3_z[i];

      p1_prime_x[i] = p1_prime_x[i] + delta_t * v1_prime_x[i];
      p1_prime_y[i] = p1_prime_y[i] + delta_t * v1_prime_y[i];
      p1_prime_z[i] = p1_prime_z[i] + delta_t * v1_prime_z[i];

      p2_prime_x[i] = p2_prime_x[i] + delta_t * v2_prime_x[i];
      p2_prime_y[i] = p2_prime_y[i] + delta_t * v2_prime_y[i];
      p2_prime_z[i] = p2_prime_z[i] + delta_t * v2_prime_z[i];

      p3_prime_x[i] = p3_prime_x[i] + delta_t * v3_prime_x[i];
      p3_prime_y[i] = p3_prime_y[i] + delta_t * v3_prime_y[i];
      p3_prime_z[i] = p3_prime_z[i] + delta_t * v3_prime_z[i];

      // assign new velocities to current velocities
      v1_x[i] = nv1_x[i];
      v1_y[i] = nv1_y[i];
      v1_z[i] = nv1_z[i];

      v2_x[i] = nv2_x[i];
      v2_y[i] = nv2_y[i];
      v2_z[i] = nv2_z[i];

      v3_x[i] = nv3_x[i];
      v3_y[i] = nv3_y[i];
      v3_z[i] = nv3_z[i];

      v1_prime_x[i] = nv1_prime_x[i];
      v1_prime_y[i] = nv1_prime_y[i];
      v1_prime_z[i] = nv1_prime_z[i];

      v2_prime_x[i] = nv2_prime_x[i];
      v2_prime_y[i] = nv2_prime_y[i];
      v2_prime_z[i] = nv2_prime_z[i];

      v3_prime_x[i] = nv3_prime_x[i];
      v3_prime_y[i] = nv3_prime_y[i];
      v3_prime_z[i] = nv3_prime_z[i];
      }
    }
  }

int main(void)
{
  // initialize timer
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  int N = 9000;
  int steps = 50000;
  double delta_t = 0.001;
  double critical_distance = 0.5;
  double m1 = 10.;
  double m2 = 20.;
  double m3 = 30.;
  double *p1_x, *p1_y, *p1_z;
  double *p2_x, *p2_y, *p2_z;
  double *p3_x, *p3_y, *p3_z;
  double *p1_prime_x, *p1_prime_y, *p1_prime_z;
  double *p2_prime_x, *p2_prime_y, *p2_prime_z;
  double *p3_prime_x, *p3_prime_y, *p3_prime_z;
  double *dv_1_x, *dv_1_y, *dv_1_z;
  double *dv_2_x, *dv_2_y, *dv_2_z;
  double *dv_3_x, *dv_3_y, *dv_3_z;
  double *dv_1pr_x, *dv_1pr_y, *dv_1pr_z;
  double *dv_2pr_x, *dv_2pr_y, *dv_2pr_z;
  double *dv_3pr_x, *dv_3pr_y, *dv_3pr_z;
  double *v1_x, *v1_y, *v1_z;
  double *v2_x, *v2_y, *v2_z;
  double *v3_x, *v3_y, *v3_z;
  double *v1_prime_x, *v1_prime_y, *v1_prime_z;
  double *v2_prime_x, *v2_prime_y, *v2_prime_z;
  double *v3_prime_x, *v3_prime_y, *v3_prime_z;
  double *nv1_x, *nv1_y, *nv1_z;
  double *nv2_x, *nv2_y, *nv2_z;
  double *nv3_x, *nv3_y, *nv3_z;
  double *nv1_prime_x, *nv1_prime_y, *nv1_prime_z;
  double *nv2_prime_x, *nv2_prime_y, *nv2_prime_z;
  double *nv3_prime_x, *nv3_prime_y, *nv3_prime_z;

  double *d_p1_x, *d_p1_y, *d_p1_z;
  double *d_p2_x, *d_p2_y, *d_p2_z;
  double *d_p3_x, *d_p3_y, *d_p3_z;
  double *d_p1_prime_x, *d_p1_prime_y, *d_p1_prime_z;
  double *d_p2_prime_x, *d_p2_prime_y, *d_p2_prime_z;
  double *d_p3_prime_x, *d_p3_prime_y, *d_p3_prime_z;
  double *d_dv_1_x, *d_dv_1_y, *d_dv_1_z;
  double *d_dv_2_x, *d_dv_2_y, *d_dv_2_z;
  double *d_dv_3_x, *d_dv_3_y, *d_dv_3_z;
  double *d_dv_1pr_x, *d_dv_1pr_y, *d_dv_1pr_z;
  double *d_dv_2pr_x, *d_dv_2pr_y, *d_dv_2pr_z;
  double *d_dv_3pr_x, *d_dv_3pr_y, *d_dv_3pr_z;
  double *d_v1_x, *d_v1_y, *d_v1_z;
  double *d_v2_x, *d_v2_y, *d_v2_z;
  double *d_v3_x, *d_v3_y, *d_v3_z;
  double *d_v1_prime_x, *d_v1_prime_y, *d_v1_prime_z;
  double *d_v2_prime_x, *d_v2_prime_y, *d_v2_prime_z;
  double *d_v3_prime_x, *d_v3_prime_y, *d_v3_prime_z;
  double *d_nv1_x, *d_nv1_y, *d_nv1_z;
  double *d_nv2_x, *d_nv2_y, *d_nv2_z;
  double *d_nv3_x, *d_nv3_y, *d_nv3_z;
  double *d_nv1_prime_x, *d_nv1_prime_y, *d_nv1_prime_z;
  double *d_nv2_prime_x, *d_nv2_prime_y, *d_nv2_prime_z;
  double *d_nv3_prime_x, *d_nv3_prime_y, *d_nv3_prime_z;

  bool *still_together, *d_still_together;
  int *times, *d_times;
  bool *not_diverged, *d_not_diverged;
  int n_gpus=4;
//  cudaGetDeviceCount(&n_gpus);
  std::cout << n_gpus << " GPUs present. Allocating CPU memory and initializing values.\n";

  cudaHostAlloc((void**)&p1_x, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc((void**)&p1_y, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc((void**)&p1_z, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);

  cudaHostAlloc((void**)&p2_x, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc((void**)&p2_y, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc((void**)&p2_z, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);

  cudaHostAlloc((void**)&p3_x, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc((void**)&p3_y, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc((void**)&p3_z, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);

  cudaHostAlloc((void**)&p1_prime_x, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc((void**)&p1_prime_y, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc((void**)&p1_prime_z, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);

  cudaHostAlloc((void**)&p2_prime_x, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc((void**)&p2_prime_y, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc((void**)&p2_prime_z, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);

  cudaHostAlloc((void**)&p3_prime_x, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc((void**)&p3_prime_y, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc((void**)&p3_prime_z, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);

  cudaHostAlloc((void**)&dv_1_x, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc((void**)&dv_1_y, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc((void**)&dv_1_z, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);

  cudaHostAlloc((void**)&dv_2_x, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc((void**)&dv_2_y, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc((void**)&dv_2_z, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);

  cudaHostAlloc((void**)&dv_3_x, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc((void**)&dv_3_y, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc((void**)&dv_3_z, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);

  cudaHostAlloc((void**)&dv_1pr_x, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc((void**)&dv_1pr_y, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc((void**)&dv_1pr_z, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);

  cudaHostAlloc((void**)&dv_2pr_x, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc((void**)&dv_2pr_y, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc((void**)&dv_2pr_z, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);

  cudaHostAlloc((void**)&dv_3pr_x, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc((void**)&dv_3pr_y, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc((void**)&dv_3pr_z, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);

  cudaHostAlloc((void**)&v1_x, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc((void**)&v1_y, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc((void**)&v1_z, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);

  cudaHostAlloc((void**)&v2_x, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc((void**)&v2_y, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc((void**)&v2_z, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);

  cudaHostAlloc((void**)&v3_x, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc((void**)&v3_y, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc((void**)&v3_z, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);

  cudaHostAlloc((void**)&v1_prime_x, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);  
  cudaHostAlloc((void**)&v1_prime_y, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc((void**)&v1_prime_z, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);

  cudaHostAlloc((void**)&v2_prime_x, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);  
  cudaHostAlloc((void**)&v2_prime_y, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc((void**)&v2_prime_z, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);

  cudaHostAlloc((void**)&v3_prime_x, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);  
  cudaHostAlloc((void**)&v3_prime_y, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc((void**)&v3_prime_z, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);

  cudaHostAlloc((void**)&nv1_x, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc((void**)&nv1_y, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc((void**)&nv1_z, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);

  cudaHostAlloc((void**)&nv2_x, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc((void**)&nv2_y, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc((void**)&nv2_z, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);

  cudaHostAlloc((void**)&nv3_x, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc((void**)&nv3_y, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc((void**)&nv3_z, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);

  cudaHostAlloc((void**)&nv1_prime_x, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc((void**)&nv1_prime_y, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc((void**)&nv1_prime_z, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);

  cudaHostAlloc((void**)&nv2_prime_x, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc((void**)&nv2_prime_y, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc((void**)&nv2_prime_z, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  
  cudaHostAlloc((void**)&nv3_prime_x, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc((void**)&nv3_prime_y, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc((void**)&nv3_prime_z, N*sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped);

  cudaHostAlloc((void**)&still_together, N*sizeof(bool), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc((void**)&not_diverged, N*sizeof(bool), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc((void**)&times, N*sizeof(int), cudaHostAllocWriteCombined | cudaHostAllocMapped);

  int resolution = sqrt(N);
  double range = 40;
  double step_size = range / resolution;
  for (int i = 0; i < N; i++) {
    int remainder = i % resolution;
    int step = i / resolution;
    p1_x[i] = -20. + 40*(double(remainder)/double(resolution));
    p1_y[i] = -20. + 40*(double(step)/double(resolution));
    p1_z[i] = -11.0;

    p2_x[i] = 0.0;
    p2_y[i] = 0.0;
    p2_z[i] = 0.0;

    p3_x[i] = 10.0;
    p3_y[i] = 10.0;
    p3_z[i] = 12.0;

    // shift p1 in all x, y, z vectors
    p1_prime_x[i] = -20. + 40*(double(remainder)/double(resolution)) + 0.001;
    p1_prime_y[i] = -20. + 40*(double(step)/double(resolution)) + 0.001;
    p1_prime_z[i] = -11.0 + 0.001;

    p2_prime_x[i] = 0.0;
    p2_prime_y[i] = 0.0;
    p2_prime_z[i] = 0.0;

    p3_prime_x[i] = 10.0;
    p3_prime_y[i] = 10.0;
    p3_prime_z[i] = 12.0;

    v1_x[i] = -3.;
    v1_y[i] = 0.;
    v1_z[i] = 0.;

    v2_x[i] = 0.;
    v2_y[i] = 0.;
    v2_z[i] = 0.;

    v3_x[i] = 3.;
    v3_y[i] = 0.;
    v3_z[i] = 0.;

    v1_prime_x[i] = -3.;
    v1_prime_y[i] = 0.;
    v1_prime_z[i] = 0.;

    v2_prime_x[i] = 0.;
    v2_prime_y[i] = 0.;
    v2_prime_z[i] = 0.;

    v3_prime_x[i] = 3.;
    v3_prime_y[i] = 0.;
    v3_prime_z[i] = 0.;

    times[i] = 0;
    still_together[i] = true;
    not_diverged[i] = true;
  }

  float *x, *d_x;

//  int N = 10000000;
  cudaHostAlloc((void**)&x, N*sizeof(float), cudaHostAllocWriteCombined | cudaHostAllocMapped);

//  int n_gpus=4;
  for (int i = 0; i < N/n_gpus; i++) {
      x[i] = 1.0f;
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
    int block_n = N/n_gpus;
    cudaSetDevice(omp_get_thread_num());
    cudaStreamCreate(&streams[d]);

    cudaMalloc(&d_p1_x, block_n*sizeof(double)); 
    cudaMalloc(&d_p1_y, block_n*sizeof(double)); 
    cudaMalloc(&d_p1_z, block_n*sizeof(double)); 

    cudaMalloc(&d_p2_x, block_n*sizeof(double));
    cudaMalloc(&d_p2_y, block_n*sizeof(double));
    cudaMalloc(&d_p2_z, block_n*sizeof(double));

    cudaMalloc(&d_p3_x, block_n*sizeof(double));
    cudaMalloc(&d_p3_y, block_n*sizeof(double));
    cudaMalloc(&d_p3_z, block_n*sizeof(double));

    cudaMalloc(&d_p1_prime_x, block_n*sizeof(double));
    cudaMalloc(&d_p1_prime_y, block_n*sizeof(double));
    cudaMalloc(&d_p1_prime_z, block_n*sizeof(double));

    cudaMalloc(&d_p2_prime_x, block_n*sizeof(double));
    cudaMalloc(&d_p2_prime_y, block_n*sizeof(double));
    cudaMalloc(&d_p2_prime_z, block_n*sizeof(double));

    cudaMalloc(&d_p3_prime_x, block_n*sizeof(double));
    cudaMalloc(&d_p3_prime_y, block_n*sizeof(double));
    cudaMalloc(&d_p3_prime_z, block_n*sizeof(double));

    cudaMalloc(&d_dv_1_x, block_n*sizeof(double));
    cudaMalloc(&d_dv_1_y, block_n*sizeof(double));
    cudaMalloc(&d_dv_1_z, block_n*sizeof(double));

    cudaMalloc(&d_dv_2_x, block_n*sizeof(double));
    cudaMalloc(&d_dv_2_y, block_n*sizeof(double));
    cudaMalloc(&d_dv_2_z, block_n*sizeof(double));

    cudaMalloc(&d_dv_3_x, block_n*sizeof(double));
    cudaMalloc(&d_dv_3_y, block_n*sizeof(double));
    cudaMalloc(&d_dv_3_z, block_n*sizeof(double));

    cudaMalloc(&d_dv_1pr_x, block_n*sizeof(double));
    cudaMalloc(&d_dv_1pr_y, block_n*sizeof(double));
    cudaMalloc(&d_dv_1pr_z, block_n*sizeof(double));

    cudaMalloc(&d_dv_2pr_x, block_n*sizeof(double));
    cudaMalloc(&d_dv_2pr_y, block_n*sizeof(double));
    cudaMalloc(&d_dv_2pr_z, block_n*sizeof(double));

    cudaMalloc(&d_dv_3pr_x, block_n*sizeof(double));
    cudaMalloc(&d_dv_3pr_y, block_n*sizeof(double));
    cudaMalloc(&d_dv_3pr_z, block_n*sizeof(double));

    cudaMalloc(&d_v1_x, block_n*sizeof(double));
    cudaMalloc(&d_v1_y, block_n*sizeof(double));
    cudaMalloc(&d_v1_z, block_n*sizeof(double));

    cudaMalloc(&d_v2_x, block_n*sizeof(double));
    cudaMalloc(&d_v2_y, block_n*sizeof(double));
    cudaMalloc(&d_v2_z, block_n*sizeof(double));

    cudaMalloc(&d_v3_x, block_n*sizeof(double));
    cudaMalloc(&d_v3_y, block_n*sizeof(double));
    cudaMalloc(&d_v3_z, block_n*sizeof(double));

    cudaMalloc(&d_v1_prime_x, block_n*sizeof(double));
    cudaMalloc(&d_v1_prime_y, block_n*sizeof(double));
    cudaMalloc(&d_v1_prime_z, block_n*sizeof(double));

    cudaMalloc(&d_v2_prime_x, block_n*sizeof(double));
    cudaMalloc(&d_v2_prime_y, block_n*sizeof(double));
    cudaMalloc(&d_v2_prime_z, block_n*sizeof(double));

    cudaMalloc(&d_v3_prime_x, block_n*sizeof(double));
    cudaMalloc(&d_v3_prime_y, block_n*sizeof(double));
    cudaMalloc(&d_v3_prime_z, block_n*sizeof(double));

    cudaMalloc(&d_nv1_x, block_n*sizeof(double));
    cudaMalloc(&d_nv1_y, block_n*sizeof(double));
    cudaMalloc(&d_nv1_z, block_n*sizeof(double));

    cudaMalloc(&d_nv2_x, block_n*sizeof(double));
    cudaMalloc(&d_nv2_y, block_n*sizeof(double));
    cudaMalloc(&d_nv2_z, block_n*sizeof(double));

    cudaMalloc(&d_nv3_x, block_n*sizeof(double));
    cudaMalloc(&d_nv3_y, block_n*sizeof(double));
    cudaMalloc(&d_nv3_z, block_n*sizeof(double));

    cudaMalloc(&d_nv1_prime_x, block_n*sizeof(double));
    cudaMalloc(&d_nv1_prime_y, block_n*sizeof(double));
    cudaMalloc(&d_nv1_prime_z, block_n*sizeof(double));

    cudaMalloc(&d_nv2_prime_x, block_n*sizeof(double));
    cudaMalloc(&d_nv2_prime_y, block_n*sizeof(double));
    cudaMalloc(&d_nv2_prime_z, block_n*sizeof(double));

    cudaMalloc(&d_nv3_prime_x, block_n*sizeof(double));
    cudaMalloc(&d_nv3_prime_y, block_n*sizeof(double));
    cudaMalloc(&d_nv3_prime_z, block_n*sizeof(double));

    cudaMalloc(&d_still_together, block_n*sizeof(bool));
    cudaMalloc(&d_times, block_n*sizeof(int));
    cudaMalloc(&d_not_diverged, block_n*sizeof(bool));

    cudaMemcpyAsync(d_p1_x, p1_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_p1_y, p1_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_p1_z, p1_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);

    cudaMemcpyAsync(d_p2_x, p2_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_p2_y, p2_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_p2_z, p2_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);

    cudaMemcpyAsync(d_p3_x, p3_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_p3_y, p3_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_p3_z, p3_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);

    cudaMemcpyAsync(d_p1_prime_x, p1_prime_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_p1_prime_y, p1_prime_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_p1_prime_z, p1_prime_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);

    cudaMemcpyAsync(d_p2_prime_x, p2_prime_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_p2_prime_y, p2_prime_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_p2_prime_z, p2_prime_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);

    cudaMemcpyAsync(d_p3_prime_x, p3_prime_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_p3_prime_y, p3_prime_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_p3_prime_z, p3_prime_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);

    cudaMemcpyAsync(d_dv_1_x, dv_1_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_dv_1_y, dv_1_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_dv_1_z, dv_1_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);

    cudaMemcpyAsync(d_dv_2_x, dv_2_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_dv_2_y, dv_2_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_dv_2_z, dv_2_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);

    cudaMemcpyAsync(d_dv_3_x, dv_3_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_dv_3_y, dv_3_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_dv_3_z, dv_3_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);

    cudaMemcpyAsync(d_dv_1pr_x, dv_1pr_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_dv_1pr_y, dv_1pr_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_dv_1pr_z, dv_1pr_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);

    cudaMemcpyAsync(d_dv_2pr_x, dv_2pr_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_dv_2pr_y, dv_2pr_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_dv_2pr_z, dv_2pr_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);

    cudaMemcpyAsync(d_dv_3pr_x, dv_3pr_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_dv_3pr_y, dv_3pr_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_dv_3pr_z, dv_3pr_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);

    cudaMemcpyAsync(d_v1_x, v1_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_v1_y, v1_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_v1_z, v1_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);

    cudaMemcpyAsync(d_v2_x, v2_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_v2_y, v2_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_v2_z, v2_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);

    cudaMemcpyAsync(d_v3_x, v3_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_v3_y, v3_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_v3_z, v3_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);

    cudaMemcpyAsync(d_v1_prime_x, v1_prime_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_v1_prime_y, v1_prime_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_v1_prime_z, v1_prime_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);

    cudaMemcpyAsync(d_v2_prime_x, v2_prime_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_v2_prime_y, v2_prime_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_v2_prime_z, v2_prime_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);

    cudaMemcpyAsync(d_v3_prime_x, v3_prime_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_v3_prime_y, v3_prime_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_v3_prime_z, v3_prime_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);

    cudaMemcpyAsync(d_nv1_x, nv1_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_nv1_y, nv1_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_nv1_z, nv1_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);

    cudaMemcpyAsync(d_nv2_x, nv2_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_nv2_y, nv2_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_nv2_z, nv2_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);

    cudaMemcpyAsync(d_nv3_x, nv3_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_nv3_y, nv3_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_nv3_z, nv3_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);

    cudaMemcpyAsync(d_nv1_prime_x, nv1_prime_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_nv1_prime_y, nv1_prime_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_nv1_prime_z, nv1_prime_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);

    cudaMemcpyAsync(d_nv2_prime_x, nv2_prime_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_nv2_prime_y, nv2_prime_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_nv2_prime_z, nv2_prime_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);

    cudaMemcpyAsync(d_nv3_prime_x, nv3_prime_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_nv3_prime_y, nv3_prime_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_nv3_prime_z, nv3_prime_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice, streams[d]);

    cudaMemcpyAsync(d_times, times+start_idx, block_n*sizeof(int), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_still_together, still_together+start_idx, block_n*sizeof(bool), cudaMemcpyHostToDevice, streams[d]);
    cudaMemcpyAsync(d_not_diverged, not_diverged+start_idx, block_n*sizeof(bool), cudaMemcpyHostToDevice, streams[d]);

//    cudaMalloc(&d_x, (N/n_gpus)*sizeof(float));
//    cudaMemcpyAsync(d_x, x+start_idx, (N/n_gpus)*sizeof(float), cudaMemcpyHostToDevice, streams[d]);

//    saxpy<<<(N+127)/128, 128, 0, streams[d]>>>(N/n_gpus, 2.0f, d_x);

//    cudaMemcpyAsync(x+start_idx, d_x, (N/n_gpus)*sizeof(float), cudaMemcpyDeviceToHost, streams[d]);
//    cudaDeviceSynchronize();

     // call CUDA kernal on inputs in configuration <<< blockIdx, threadIdx, 0, stream>>>>
     divergence<<<(block_n+255)/256, 256, 0, streams[d]>>>(
         block_n, 
         steps, 
         delta_t,
         d_still_together,
         d_not_diverged,
         d_times,
         m1, m2, m3,
         critical_distance,
         d_p1_x, d_p1_y, d_p1_z, 
         d_p2_x, d_p2_y, d_p2_z, 
        d_p3_x, d_p3_y, d_p3_z, 
         d_p1_prime_x, d_p1_prime_y, d_p1_prime_z, 
        d_p2_prime_x, d_p2_prime_y, d_p2_prime_z, 
         d_p3_prime_x, d_p3_prime_y, d_p3_prime_z,
         d_dv_1_x, d_dv_1_y, d_dv_1_z,
         d_dv_2_x, d_dv_2_y, d_dv_2_z,
        d_dv_3_x, d_dv_3_y, d_dv_3_z,
         d_dv_1pr_x, d_dv_1pr_y, d_dv_1pr_z,
        d_dv_2pr_x, d_dv_2pr_y, d_dv_2pr_z,
         d_dv_3pr_x, d_dv_3pr_y, d_dv_3pr_z,
         d_v1_x, d_v1_y, d_v1_z,
         d_v2_x, d_v2_y, d_v2_z,
         d_v3_x, d_v3_y, d_v3_z,
         d_v1_prime_x, d_v1_prime_y, d_v1_prime_z,
         d_v2_prime_x, d_v2_prime_y, d_v2_prime_z,
         d_v3_prime_x, d_v3_prime_y, d_v3_prime_z,
         d_nv1_x, d_nv1_y, d_nv1_z,
         d_nv2_x, d_nv2_y, d_nv2_z,
        d_nv3_x, d_nv3_y, d_nv3_z,
         d_nv1_prime_x, d_nv1_prime_y, d_nv1_prime_z,    
         d_nv2_prime_x, d_nv2_prime_y, d_nv2_prime_z,
        d_nv3_prime_x, d_nv3_prime_y, d_nv3_prime_z
         );

    cudaMemcpyAsync(times+start_idx, d_times, block_n*sizeof(int), cudaMemcpyDeviceToHost, streams[d]);
    // cudaMemcpyAsync(still_together+start_idx, d_still_together, block_n*sizeof(bool), cudaMemcpyDeviceToHost, streams[d]);
    // cudaMemcpyAsync(not_diverged+start_idx, d_not_diverged, block_n*sizeof(bool), cudaMemcpyDeviceToHost, streams[d]);
    // cudaMemcpyAsync(p1_x+start_idx, d_p1_x, block_n*sizeof(double), cudaMemcpyDeviceToHost, streams[d]);
    // cudaMemcpyAsync(p1_y+start_idx, d_p1_y, block_n*sizeof(double), cudaMemcpyDeviceToHost, streams[d]);
    // cudaMemcpyAsync(p1_z+start_idx, d_p1_z, block_n*sizeof(double), cudaMemcpyDeviceToHost, streams[d]);
    // cudaMemcpyAsync(p1_prime_x+start_idx, d_p1_prime_x, block_n*sizeof(double), cudaMemcpyDeviceToHost, streams[d]);
    // cudaMemcpyAsync(p1_prime_y+start_idx, d_p1_prime_y, block_n*sizeof(double), cudaMemcpyDeviceToHost, streams[d]);
    // cudaMemcpyAsync(p1_prime_z+start_idx, d_p1_prime_z, block_n*sizeof(double), cudaMemcpyDeviceToHost, streams[d]);
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
  std::cout << x[0]<< "\n";
  cudaError_t err = cudaGetLastError();  // add
std::cout << err << "\n";
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

