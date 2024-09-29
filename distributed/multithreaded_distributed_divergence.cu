#include <stdio.h>
#include <iostream>
#include <chrono>
#include <omp.h>
#include <thread>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

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
  int n_gpus;
  cudaGetDeviceCount(&n_gpus);
  std::cout << n_gpus << " GPUs present. Allocating CPU memory and initializing values.\n";
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

  double * d_p1_x[n_gpus], * d_p1_y[n_gpus], * d_p1_z[n_gpus];
  double * d_p2_x[n_gpus], * d_p2_y[n_gpus], * d_p2_z[n_gpus];
  double * d_p3_x[n_gpus], * d_p3_y[n_gpus], * d_p3_z[n_gpus];
  double * d_p1_prime_x[n_gpus], * d_p1_prime_y[n_gpus], * d_p1_prime_z[n_gpus];
  double * d_p2_prime_x[n_gpus], * d_p2_prime_y[n_gpus], * d_p2_prime_z[n_gpus];
  double * d_p3_prime_x[n_gpus], * d_p3_prime_y[n_gpus], * d_p3_prime_z[n_gpus];
  double * d_dv_1_x[n_gpus], * d_dv_1_y[n_gpus], * d_dv_1_z[n_gpus];
  double * d_dv_2_x[n_gpus], * d_dv_2_y[n_gpus], * d_dv_2_z[n_gpus];
  double * d_dv_3_x[n_gpus], * d_dv_3_y[n_gpus], * d_dv_3_z[n_gpus];
  double * d_dv_1pr_x[n_gpus], * d_dv_1pr_y[n_gpus], * d_dv_1pr_z[n_gpus];
  double * d_dv_2pr_x[n_gpus], * d_dv_2pr_y[n_gpus], * d_dv_2pr_z[n_gpus];
  double * d_dv_3pr_x[n_gpus], * d_dv_3pr_y[n_gpus], * d_dv_3pr_z[n_gpus];
  double * d_v1_x[n_gpus], * d_v1_y[n_gpus], * d_v1_z[n_gpus];
  double * d_v2_x[n_gpus], * d_v2_y[n_gpus], * d_v2_z[n_gpus];
  double * d_v3_x[n_gpus], * d_v3_y[n_gpus], * d_v3_z[n_gpus];
  double * d_v1_prime_x[n_gpus], * d_v1_prime_y[n_gpus], * d_v1_prime_z[n_gpus];
  double * d_v2_prime_x[n_gpus], * d_v2_prime_y[n_gpus], * d_v2_prime_z[n_gpus];
  double * d_v3_prime_x[n_gpus], * d_v3_prime_y[n_gpus], * d_v3_prime_z[n_gpus];
  double * d_nv1_x[n_gpus], * d_nv1_y[n_gpus], * d_nv1_z[n_gpus];
  double * d_nv2_x[n_gpus], * d_nv2_y[n_gpus], * d_nv2_z[n_gpus];
  double * d_nv3_x[n_gpus], * d_nv3_y[n_gpus], * d_nv3_z[n_gpus];
  double * d_nv1_prime_x[n_gpus], * d_nv1_prime_y[n_gpus], * d_nv1_prime_z[n_gpus];
  double * d_nv2_prime_x[n_gpus], * d_nv2_prime_y[n_gpus], * d_nv2_prime_z[n_gpus];
  double * d_nv3_prime_x[n_gpus], * d_nv3_prime_y[n_gpus], * d_nv3_prime_z[n_gpus];

  bool *still_together, * d_still_together[n_gpus];
  int *times, * d_times[n_gpus];
  bool *not_diverged, * d_not_diverged[n_gpus];

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

  // launch one thread per GPU
  cudaStream_t streams[n_gpus];
  #pragma omp parallel num_threads(n_gpus)
  {
    int d = omp_get_thread_num();
    std::cout << "Thread " << d << " initialized \n";
    // assumes that n_gpus divides N with no remainder, which is safe as N is a large square.
    int start_idx = (N/n_gpus)*d;
    int end_idx = start_idx + N/n_gpus;
    std::cout << "Start index: " << start_idx << "\nEnd index: " << end_idx << "\n";
    int block_n = N/n_gpus;
    cudaSetDevice(omp_get_thread_num());
    cudaStreamCreate(&streams[d]);

    cudaMalloc(&d_p1_x[d], block_n*sizeof(double)); 
    cudaMalloc(&d_p1_y[d], block_n*sizeof(double)); 
    cudaMalloc(&d_p1_z[d], block_n*sizeof(double)); 

    cudaMalloc(&d_p2_x[d], block_n*sizeof(double));
    cudaMalloc(&d_p2_y[d], block_n*sizeof(double));
    cudaMalloc(&d_p2_z[d], block_n*sizeof(double));

    cudaMalloc(&d_p3_x[d], block_n*sizeof(double));
    cudaMalloc(&d_p3_y[d], block_n*sizeof(double));
    cudaMalloc(&d_p3_z[d], block_n*sizeof(double));

    cudaMalloc(&d_p1_prime_x[d], block_n*sizeof(double));
    cudaMalloc(&d_p1_prime_y[d], block_n*sizeof(double));
    cudaMalloc(&d_p1_prime_z[d], block_n*sizeof(double));

    cudaMalloc(&d_p2_prime_x[d], block_n*sizeof(double));
    cudaMalloc(&d_p2_prime_y[d], block_n*sizeof(double));
    cudaMalloc(&d_p2_prime_z[d], block_n*sizeof(double));

    cudaMalloc(&d_p3_prime_x[d], block_n*sizeof(double));
    cudaMalloc(&d_p3_prime_y[d], block_n*sizeof(double));
    cudaMalloc(&d_p3_prime_z[d], block_n*sizeof(double));

    cudaMalloc(&d_dv_1_x[d], block_n*sizeof(double));
    cudaMalloc(&d_dv_1_y[d], block_n*sizeof(double));
    cudaMalloc(&d_dv_1_z[d], block_n*sizeof(double));

    cudaMalloc(&d_dv_2_x[d], block_n*sizeof(double));
    cudaMalloc(&d_dv_2_y[d], block_n*sizeof(double));
    cudaMalloc(&d_dv_2_z[d], block_n*sizeof(double));

    cudaMalloc(&d_dv_3_x[d], block_n*sizeof(double));
    cudaMalloc(&d_dv_3_y[d], block_n*sizeof(double));
    cudaMalloc(&d_dv_3_z[d], block_n*sizeof(double));

    cudaMalloc(&d_dv_1pr_x[d], block_n*sizeof(double));
    cudaMalloc(&d_dv_1pr_y[d], block_n*sizeof(double));
    cudaMalloc(&d_dv_1pr_z[d], block_n*sizeof(double));

    cudaMalloc(&d_dv_2pr_x[d], block_n*sizeof(double));
    cudaMalloc(&d_dv_2pr_y[d], block_n*sizeof(double));
    cudaMalloc(&d_dv_2pr_z[d], block_n*sizeof(double));

    cudaMalloc(&d_dv_3pr_x[d], block_n*sizeof(double));
    cudaMalloc(&d_dv_3pr_y[d], block_n*sizeof(double));
    cudaMalloc(&d_dv_3pr_z[d], block_n*sizeof(double));

    cudaMalloc(&d_v1_x[d], block_n*sizeof(double));
    cudaMalloc(&d_v1_y[d], block_n*sizeof(double));
    cudaMalloc(&d_v1_z[d], block_n*sizeof(double));

    cudaMalloc(&d_v2_x[d], block_n*sizeof(double));
    cudaMalloc(&d_v2_y[d], block_n*sizeof(double));
    cudaMalloc(&d_v2_z[d], block_n*sizeof(double));

    cudaMalloc(&d_v3_x[d], block_n*sizeof(double));
    cudaMalloc(&d_v3_y[d], block_n*sizeof(double));
    cudaMalloc(&d_v3_z[d], block_n*sizeof(double));

    cudaMalloc(&d_v1_prime_x[d], block_n*sizeof(double));
    cudaMalloc(&d_v1_prime_y[d], block_n*sizeof(double));
    cudaMalloc(&d_v1_prime_z[d], block_n*sizeof(double));

    cudaMalloc(&d_v2_prime_x[d], block_n*sizeof(double));
    cudaMalloc(&d_v2_prime_y[d], block_n*sizeof(double));
    cudaMalloc(&d_v2_prime_z[d], block_n*sizeof(double));

    cudaMalloc(&d_v3_prime_x[d], block_n*sizeof(double));
    cudaMalloc(&d_v3_prime_y[d], block_n*sizeof(double));
    cudaMalloc(&d_v3_prime_z[d], block_n*sizeof(double));

    cudaMalloc(&d_nv1_x[d], block_n*sizeof(double));
    cudaMalloc(&d_nv1_y[d], block_n*sizeof(double));
    cudaMalloc(&d_nv1_z[d], block_n*sizeof(double));

    cudaMalloc(&d_nv2_x[d], block_n*sizeof(double));
    cudaMalloc(&d_nv2_y[d], block_n*sizeof(double));
    cudaMalloc(&d_nv2_z[d], block_n*sizeof(double));

    cudaMalloc(&d_nv3_x[d], block_n*sizeof(double));
    cudaMalloc(&d_nv3_y[d], block_n*sizeof(double));
    cudaMalloc(&d_nv3_z[d], block_n*sizeof(double));

    cudaMalloc(&d_nv1_prime_x[d], block_n*sizeof(double));
    cudaMalloc(&d_nv1_prime_y[d], block_n*sizeof(double));
    cudaMalloc(&d_nv1_prime_z[d], block_n*sizeof(double));

    cudaMalloc(&d_nv2_prime_x[d], block_n*sizeof(double));
    cudaMalloc(&d_nv2_prime_y[d], block_n*sizeof(double));
    cudaMalloc(&d_nv2_prime_z[d], block_n*sizeof(double));

    cudaMalloc(&d_nv3_prime_x[d], block_n*sizeof(double));
    cudaMalloc(&d_nv3_prime_y[d], block_n*sizeof(double));
    cudaMalloc(&d_nv3_prime_z[d], block_n*sizeof(double));

    cudaMalloc(&d_still_together[d], block_n*sizeof(bool));
    cudaMalloc(&d_times[d], block_n*sizeof(int));
    cudaMalloc(&d_not_diverged[d], block_n*sizeof(bool));

    cudaMemcpy(d_p1_x[d], p1_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p1_y[d], p1_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p1_z[d], p1_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_p2_x[d], p2_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p2_y[d], p2_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p2_z[d], p2_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_p3_x[d], p3_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p3_y[d], p3_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p3_z[d], p3_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_p1_prime_x[d], p1_prime_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p1_prime_y[d], p1_prime_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p1_prime_z[d], p1_prime_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_p2_prime_x[d], p2_prime_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p2_prime_y[d], p2_prime_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p2_prime_z[d], p2_prime_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_p3_prime_x[d], p3_prime_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p3_prime_y[d], p3_prime_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p3_prime_z[d], p3_prime_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_dv_1_x[d], dv_1_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dv_1_y[d], dv_1_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dv_1_z[d], dv_1_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_dv_2_x[d], dv_2_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dv_2_y[d], dv_2_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dv_2_z[d], dv_2_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_dv_3_x[d], dv_3_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dv_3_y[d], dv_3_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dv_3_z[d], dv_3_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_dv_1pr_x[d], dv_1pr_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dv_1pr_y[d], dv_1pr_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dv_1pr_z[d], dv_1pr_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_dv_2pr_x[d], dv_2pr_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dv_2pr_y[d], dv_2pr_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dv_2pr_z[d], dv_2pr_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_dv_3pr_x[d], dv_3pr_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dv_3pr_y[d], dv_3pr_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dv_3pr_z[d], dv_3pr_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_v1_x[d], v1_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v1_y[d], v1_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v1_z[d], v1_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_v2_x[d], v2_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v2_y[d], v2_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v2_z[d], v2_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_v3_x[d], v3_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v3_y[d], v3_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v3_z[d], v3_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_v1_prime_x[d], v1_prime_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v1_prime_y[d], v1_prime_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v1_prime_z[d], v1_prime_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_v2_prime_x[d], v2_prime_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v2_prime_y[d], v2_prime_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v2_prime_z[d], v2_prime_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_v3_prime_x[d], v3_prime_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v3_prime_y[d], v3_prime_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v3_prime_z[d], v3_prime_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_nv1_x[d], nv1_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nv1_y[d], nv1_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nv1_z[d], nv1_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_nv2_x[d], nv2_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nv2_y[d], nv2_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nv2_z[d], nv2_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_nv3_x[d], nv3_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nv3_y[d], nv3_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nv3_z[d], nv3_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_nv1_prime_x[d], nv1_prime_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nv1_prime_y[d], nv1_prime_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nv1_prime_z[d], nv1_prime_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_nv2_prime_x[d], nv2_prime_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nv2_prime_y[d], nv2_prime_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nv2_prime_z[d], nv2_prime_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_nv3_prime_x[d], nv3_prime_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nv3_prime_y[d], nv3_prime_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nv3_prime_z[d], nv3_prime_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_times[d], times+start_idx, block_n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_still_together[d], still_together+start_idx, block_n*sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(d_not_diverged[d], not_diverged+start_idx, block_n*sizeof(bool), cudaMemcpyHostToDevice);

    // call CUDA kernal on inputs in configuration <<< blockIdx, threadIdx>>>>
    divergence<<<(block_n+127)/128, 128>>>(
        block_n, 
        steps, 
        delta_t,
        d_still_together[d],
        d_not_diverged[d],
        d_times[d],
        m1, m2, m3,
        critical_distance,
        d_p1_x[d], d_p1_y[d], d_p1_z[d], 
        d_p2_x[d], d_p2_y[d], d_p2_z[d], 
        d_p3_x[d], d_p3_y[d], d_p3_z[d], 
        d_p1_prime_x[d], d_p1_prime_y[d], d_p1_prime_z[d], 
        d_p2_prime_x[d], d_p2_prime_y[d], d_p2_prime_z[d], 
        d_p3_prime_x[d], d_p3_prime_y[d], d_p3_prime_z[d],
        d_dv_1_x[d], d_dv_1_y[d], d_dv_1_z[d],
        d_dv_2_x[d], d_dv_2_y[d], d_dv_2_z[d],
        d_dv_3_x[d], d_dv_3_y[d], d_dv_3_z[d],
        d_dv_1pr_x[d], d_dv_1pr_y[d], d_dv_1pr_z[d],
        d_dv_2pr_x[d], d_dv_2pr_y[d], d_dv_2pr_z[d],
        d_dv_3pr_x[d], d_dv_3pr_y[d], d_dv_3pr_z[d],
        d_v1_x[d], d_v1_y[d], d_v1_z[d],
        d_v2_x[d], d_v2_y[d], d_v2_z[d],
        d_v3_x[d], d_v3_y[d], d_v3_z[d],
        d_v1_prime_x[d], d_v1_prime_y[d], d_v1_prime_z[d],
        d_v2_prime_x[d], d_v2_prime_y[d], d_v2_prime_z[d],
        d_v3_prime_x[d], d_v3_prime_y[d], d_v3_prime_z[d],
        d_nv1_x[d], d_nv1_y[d], d_nv1_z[d],
        d_nv2_x[d], d_nv2_y[d], d_nv2_z[d],
        d_nv3_x[d], d_nv3_y[d], d_nv3_z[d],
        d_nv1_prime_x[d], d_nv1_prime_y[d], d_nv1_prime_z[d],
        d_nv2_prime_x[d], d_nv2_prime_y[d], d_nv2_prime_z[d],
        d_nv3_prime_x[d], d_nv3_prime_y[d], d_nv3_prime_z[d]
        );

    cudaMemcpyAsync(times+start_idx, d_times[d], block_n*sizeof(int), cudaMemcpyDeviceToHost, streams[d]);
    cudaDeviceSynchronize();
    cudaFree(d_p1_x[d]); cudaFree(d_p1_y[d]); cudaFree(d_p1_z[d]);
    cudaFree(d_p2_x[d]); cudaFree(d_p2_y[d]); cudaFree(d_p2_z[d]);
    cudaFree(d_p3_x[d]); cudaFree(d_p3_y[d]); cudaFree(d_p3_z[d]);

    cudaFree(d_p1_prime_x[d]); cudaFree(d_p1_prime_y[d]); cudaFree(d_p1_prime_z[d]);  
    cudaFree(d_p2_prime_x[d]); cudaFree(d_p2_prime_y[d]); cudaFree(d_p2_prime_z[d]);
    cudaFree(d_p3_prime_x[d]); cudaFree(d_p3_prime_y[d]); cudaFree(d_p3_prime_z[d]);

    cudaFree(d_dv_1_x[d]); cudaFree(d_dv_1_y[d]); cudaFree(d_dv_1_z[d]);
    cudaFree(d_dv_2_x[d]); cudaFree(d_dv_2_y[d]); cudaFree(d_dv_2_z[d]);
    cudaFree(d_dv_3_x[d]); cudaFree(d_dv_3_y[d]); cudaFree(d_dv_3_z[d]);

    cudaFree(d_dv_1pr_x[d]); cudaFree(d_dv_1pr_y[d]); cudaFree(d_dv_1pr_z[d]);
    cudaFree(d_dv_2pr_x[d]); cudaFree(d_dv_2pr_y[d]); cudaFree(d_dv_2pr_z[d]);
    cudaFree(d_dv_3pr_x[d]); cudaFree(d_dv_3pr_y[d]); cudaFree(d_dv_3pr_z[d]);

    cudaFree(d_nv1_x[d]); cudaFree(d_nv1_y[d]); cudaFree(d_nv1_z[d]);
    cudaFree(d_nv2_x[d]); cudaFree(d_nv2_y[d]); cudaFree(d_nv2_z[d]);
    cudaFree(d_nv3_x[d]); cudaFree(d_nv3_y[d]); cudaFree(d_nv3_z[d]);

    cudaFree(d_v1_x[d]); cudaFree(d_v1_y[d]); cudaFree(d_v1_z[d]);
    cudaFree(d_v2_x[d]); cudaFree(d_v2_y[d]); cudaFree(d_v2_z[d]);
    cudaFree(d_v3_x[d]); cudaFree(d_v3_y[d]); cudaFree(d_v3_z[d]);

    cudaFree(d_v1_prime_x[d]); cudaFree(d_v1_prime_y[d]); cudaFree(d_v1_prime_z[d]);
    cudaFree(d_v2_prime_x[d]); cudaFree(d_v2_prime_y[d]); cudaFree(d_v2_prime_z[d]);
    cudaFree(d_v3_prime_x[d]); cudaFree(d_v3_prime_y[d]); cudaFree(d_v3_prime_z[d]);

    cudaFree(d_nv1_prime_x[d]); cudaFree(d_nv1_prime_y[d]); cudaFree(d_nv1_prime_z[d]);
    cudaFree(d_nv2_prime_x[d]); cudaFree(d_nv2_prime_y[d]); cudaFree(d_nv2_prime_z[d]);
    cudaFree(d_nv3_prime_x[d]); cudaFree(d_nv3_prime_y[d]); cudaFree(d_nv3_prime_z[d]);

    cudaFree(d_times[d]); cudaFree(d_still_together[d]); cudaFree(d_not_diverged[d]);
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
  // std::cout << x[0]<< "\n";

  cudaFreeHost(p1_x); cudaFreeHost(p1_y); cudaFreeHost(p1_z);
  cudaFreeHost(p2_x); cudaFreeHost(p2_y); cudaFreeHost(p2_z);
  cudaFreeHost(p3_x); cudaFreeHost(p3_y); cudaFreeHost(p3_z);

  cudaFreeHost(p1_prime_x); cudaFreeHost(p1_prime_y); cudaFreeHost(p1_prime_z);  
  cudaFreeHost(p2_prime_x); cudaFreeHost(p2_prime_y); cudaFreeHost(p2_prime_z);  
  cudaFreeHost(p3_prime_x); cudaFreeHost(p3_prime_y); cudaFreeHost(p3_prime_z);  

  cudaFreeHost(dv_1_x); cudaFreeHost(dv_1_y); cudaFreeHost(dv_1_z);
  cudaFreeHost(dv_2_x); cudaFreeHost(dv_2_y); cudaFreeHost(dv_2_z);
  cudaFreeHost(dv_3_x); cudaFreeHost(dv_3_y); cudaFreeHost(dv_3_z);

  cudaFreeHost(dv_1pr_x); cudaFreeHost(dv_1pr_y); cudaFreeHost(dv_1pr_z);
  cudaFreeHost(dv_2pr_x); cudaFreeHost(dv_2pr_y); cudaFreeHost(dv_2pr_z);
  cudaFreeHost(dv_3pr_x); cudaFreeHost(dv_3pr_y); cudaFreeHost(dv_3pr_z);

  cudaFreeHost(v1_x); cudaFreeHost(v1_y); cudaFreeHost(v1_z);
  cudaFreeHost(v2_x); cudaFreeHost(v2_y); cudaFreeHost(v2_z);
  cudaFreeHost(v3_x); cudaFreeHost(v3_y); cudaFreeHost(v3_z);

  cudaFreeHost(v1_prime_x); cudaFreeHost(v1_prime_y); cudaFreeHost(v1_prime_z);
  cudaFreeHost(v2_prime_x); cudaFreeHost(v2_prime_y); cudaFreeHost(v2_prime_z);
  cudaFreeHost(v3_prime_x); cudaFreeHost(v3_prime_y); cudaFreeHost(v3_prime_z);

  cudaFreeHost(nv1_x); cudaFreeHost(nv1_y); cudaFreeHost(nv1_z);
  cudaFreeHost(nv2_x); cudaFreeHost(nv2_y); cudaFreeHost(nv2_z);
  cudaFreeHost(nv3_x); cudaFreeHost(nv3_y); cudaFreeHost(nv3_z);

  cudaFreeHost(nv1_prime_x); cudaFreeHost(nv1_prime_y); cudaFreeHost(nv1_prime_z);
  cudaFreeHost(nv2_prime_x); cudaFreeHost(nv2_prime_y); cudaFreeHost(nv2_prime_z);
  cudaFreeHost(nv3_prime_x); cudaFreeHost(nv3_prime_y); cudaFreeHost(nv3_prime_z);

  cudaFreeHost(still_together); cudaFreeHost(not_diverged);
  // cudaError_t err = cudaGetLastError();  // add
  // std::cout << err << "\n";
  // if (err != cudaSuccess) std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl; // add
  
  // measure elapsed kernal runtime
  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  std::cout << "Elapsed Time: " << elapsed_seconds.count() << "s\n";
  }

