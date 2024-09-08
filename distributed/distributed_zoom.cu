#include <stdio.h>
#include <iostream>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

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
extern "C" {
    int* divergence(int x_res, int y_res, int time_steps, double x_center, double x_range, double y_center, double y_range, double shift_distance)
    {
      int n_gpus;
      cudaGetDeviceCount(&n_gpus);
      std::cout << n_gpus << " GPUs present. Allocating CPU memory and initializing values.";

      int N = x_res * y_res;
      int steps = time_steps;
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

      // initialize timer
      std::chrono::time_point<std::chrono::system_clock> start, end;
      start = std::chrono::system_clock::now();

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

      double start_x = x_center - x_range/2;
      double start_y = y_center - y_range/2;

      for (int i = 0; i < N; i++) {
        int remainder = i % y_res;
        int step = i / x_res;
        p1_x[i] = start_x + x_range*(double(remainder)/double(x_res)); // 5.3
        p1_y[i] = start_y + y_range*(double(step)/double(y_res)); // 0.45
        p1_z[i] = -11.0;

        p2_x[i] = 0.0;
        p2_y[i] = 0.0;
        p2_z[i] = 0.0;

        p3_x[i] = 10.0;
        p3_y[i] = 10.0;
        p3_z[i] = 12.0;

        // shift p1 in all x, y, z vectors
        p1_prime_x[i] = start_x + x_range*(double(remainder)/double(x_res)) + shift_distance; // 0.0000001
        p1_prime_y[i] = start_y + y_range*(double(step)/double(y_res)) + shift_distance;
        p1_prime_z[i] = -11.0 + shift_distance;

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

      // launch GPUs using one CPU thread
      for (int i=0; i<n_gpus; i++){
        std::cout << "GPU number " << i << " initialized" << "\n";
        // assumes that n_gpus divides N with no remainder, which is safe as N is a large square.
        int start_idx = (N/n_gpus)*i;
        int end_idx = start_idx + N/n_gpus;
        std::cout << "Start index: " << start_idx << "\nEnd index: " << end_idx << "\n";
        int block_n = N/n_gpus;
        cudaSetDevice(i); 
        std::cout << &d_p1_x[i];

        cudaMalloc(&d_p1_x[i], block_n*sizeof(double)); 
        cudaMalloc(&d_p1_y[i], block_n*sizeof(double)); 
        cudaMalloc(&d_p1_z[i], block_n*sizeof(double)); 

        cudaMalloc(&d_p2_x[i], block_n*sizeof(double));
        cudaMalloc(&d_p2_y[i], block_n*sizeof(double));
        cudaMalloc(&d_p2_z[i], block_n*sizeof(double));

        cudaMalloc(&d_p3_x[i], block_n*sizeof(double));
        cudaMalloc(&d_p3_y[i], block_n*sizeof(double));
        cudaMalloc(&d_p3_z[i], block_n*sizeof(double));

        cudaMalloc(&d_p1_prime_x[i], block_n*sizeof(double));
        cudaMalloc(&d_p1_prime_y[i], block_n*sizeof(double));
        cudaMalloc(&d_p1_prime_z[i], block_n*sizeof(double));

        cudaMalloc(&d_p2_prime_x[i], block_n*sizeof(double));
        cudaMalloc(&d_p2_prime_y[i], block_n*sizeof(double));
        cudaMalloc(&d_p2_prime_z[i], block_n*sizeof(double));

        cudaMalloc(&d_p3_prime_x[i], block_n*sizeof(double));
        cudaMalloc(&d_p3_prime_y[i], block_n*sizeof(double));
        cudaMalloc(&d_p3_prime_z[i], block_n*sizeof(double));

        cudaMalloc(&d_dv_1_x[i], block_n*sizeof(double));
        cudaMalloc(&d_dv_1_y[i], block_n*sizeof(double));
        cudaMalloc(&d_dv_1_z[i], block_n*sizeof(double));

        cudaMalloc(&d_dv_2_x[i], block_n*sizeof(double));
        cudaMalloc(&d_dv_2_y[i], block_n*sizeof(double));
        cudaMalloc(&d_dv_2_z[i], block_n*sizeof(double));

        cudaMalloc(&d_dv_3_x[i], block_n*sizeof(double));
        cudaMalloc(&d_dv_3_y[i], block_n*sizeof(double));
        cudaMalloc(&d_dv_3_z[i], block_n*sizeof(double));

        cudaMalloc(&d_dv_1pr_x[i], block_n*sizeof(double));
        cudaMalloc(&d_dv_1pr_y[i], block_n*sizeof(double));
        cudaMalloc(&d_dv_1pr_z[i], block_n*sizeof(double));

        cudaMalloc(&d_dv_2pr_x[i], block_n*sizeof(double));
        cudaMalloc(&d_dv_2pr_y[i], block_n*sizeof(double));
        cudaMalloc(&d_dv_2pr_z[i], block_n*sizeof(double));

        cudaMalloc(&d_dv_3pr_x[i], block_n*sizeof(double));
        cudaMalloc(&d_dv_3pr_y[i], block_n*sizeof(double));
        cudaMalloc(&d_dv_3pr_z[i], block_n*sizeof(double));

        cudaMalloc(&d_v1_x[i], block_n*sizeof(double));
        cudaMalloc(&d_v1_y[i], block_n*sizeof(double));
        cudaMalloc(&d_v1_z[i], block_n*sizeof(double));

        cudaMalloc(&d_v2_x[i], block_n*sizeof(double));
        cudaMalloc(&d_v2_y[i], block_n*sizeof(double));
        cudaMalloc(&d_v2_z[i], block_n*sizeof(double));

        cudaMalloc(&d_v3_x[i], block_n*sizeof(double));
        cudaMalloc(&d_v3_y[i], block_n*sizeof(double));
        cudaMalloc(&d_v3_z[i], block_n*sizeof(double));

        cudaMalloc(&d_v1_prime_x[i], block_n*sizeof(double));
        cudaMalloc(&d_v1_prime_y[i], block_n*sizeof(double));
        cudaMalloc(&d_v1_prime_z[i], block_n*sizeof(double));

        cudaMalloc(&d_v2_prime_x[i], block_n*sizeof(double));
        cudaMalloc(&d_v2_prime_y[i], block_n*sizeof(double));
        cudaMalloc(&d_v2_prime_z[i], block_n*sizeof(double));

        cudaMalloc(&d_v3_prime_x[i], block_n*sizeof(double));
        cudaMalloc(&d_v3_prime_y[i], block_n*sizeof(double));
        cudaMalloc(&d_v3_prime_z[i], block_n*sizeof(double));

        cudaMalloc(&d_nv1_x[i], block_n*sizeof(double));
        cudaMalloc(&d_nv1_y[i], block_n*sizeof(double));
        cudaMalloc(&d_nv1_z[i], block_n*sizeof(double));

        cudaMalloc(&d_nv2_x[i], block_n*sizeof(double));
        cudaMalloc(&d_nv2_y[i], block_n*sizeof(double));
        cudaMalloc(&d_nv2_z[i], block_n*sizeof(double));

        cudaMalloc(&d_nv3_x[i], block_n*sizeof(double));
        cudaMalloc(&d_nv3_y[i], block_n*sizeof(double));
        cudaMalloc(&d_nv3_z[i], block_n*sizeof(double));

        cudaMalloc(&d_nv1_prime_x[i], block_n*sizeof(double));
        cudaMalloc(&d_nv1_prime_y[i], block_n*sizeof(double));
        cudaMalloc(&d_nv1_prime_z[i], block_n*sizeof(double));

        cudaMalloc(&d_nv2_prime_x[i], block_n*sizeof(double));
        cudaMalloc(&d_nv2_prime_y[i], block_n*sizeof(double));
        cudaMalloc(&d_nv2_prime_z[i], block_n*sizeof(double));

        cudaMalloc(&d_nv3_prime_x[i], block_n*sizeof(double));
        cudaMalloc(&d_nv3_prime_y[i], block_n*sizeof(double));
        cudaMalloc(&d_nv3_prime_z[i], block_n*sizeof(double));

        cudaMalloc(&d_still_together[i], block_n*sizeof(bool));
        cudaMalloc(&d_times[i], block_n*sizeof(int));
        cudaMalloc(&d_not_diverged[i], block_n*sizeof(bool));

        cudaMemcpy(d_p1_x[i], p1_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_p1_y[i], p1_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_p1_z[i], p1_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);

        cudaMemcpy(d_p2_x[i], p2_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_p2_y[i], p2_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_p2_z[i], p2_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);

        cudaMemcpy(d_p3_x[i], p3_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_p3_y[i], p3_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_p3_z[i], p3_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);

        cudaMemcpy(d_p1_prime_x[i], p1_prime_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_p1_prime_y[i], p1_prime_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_p1_prime_z[i], p1_prime_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);

        cudaMemcpy(d_p2_prime_x[i], p2_prime_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_p2_prime_y[i], p2_prime_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_p2_prime_z[i], p2_prime_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);

        cudaMemcpy(d_p3_prime_x[i], p3_prime_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_p3_prime_y[i], p3_prime_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_p3_prime_z[i], p3_prime_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);

        cudaMemcpy(d_dv_1_x[i], dv_1_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_dv_1_y[i], dv_1_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_dv_1_z[i], dv_1_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);

        cudaMemcpy(d_dv_2_x[i], dv_2_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_dv_2_y[i], dv_2_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_dv_2_z[i], dv_2_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);

        cudaMemcpy(d_dv_3_x[i], dv_3_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_dv_3_y[i], dv_3_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_dv_3_z[i], dv_3_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);

        cudaMemcpy(d_dv_1pr_x[i], dv_1pr_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_dv_1pr_y[i], dv_1pr_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_dv_1pr_z[i], dv_1pr_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);

        cudaMemcpy(d_dv_2pr_x[i], dv_2pr_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_dv_2pr_y[i], dv_2pr_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_dv_2pr_z[i], dv_2pr_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);

        cudaMemcpy(d_dv_3pr_x[i], dv_3pr_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_dv_3pr_y[i], dv_3pr_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_dv_3pr_z[i], dv_3pr_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);

        cudaMemcpy(d_v1_x[i], v1_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_v1_y[i], v1_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_v1_z[i], v1_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);

        cudaMemcpy(d_v2_x[i], v2_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_v2_y[i], v2_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_v2_z[i], v2_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);

        cudaMemcpy(d_v3_x[i], v3_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_v3_y[i], v3_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_v3_z[i], v3_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);

        cudaMemcpy(d_v1_prime_x[i], v1_prime_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_v1_prime_y[i], v1_prime_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_v1_prime_z[i], v1_prime_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);

        cudaMemcpy(d_v2_prime_x[i], v2_prime_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_v2_prime_y[i], v2_prime_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_v2_prime_z[i], v2_prime_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);

        cudaMemcpy(d_v3_prime_x[i], v3_prime_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_v3_prime_y[i], v3_prime_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_v3_prime_z[i], v3_prime_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);

        cudaMemcpy(d_nv1_x[i], nv1_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_nv1_y[i], nv1_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_nv1_z[i], nv1_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);

        cudaMemcpy(d_nv2_x[i], nv2_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_nv2_y[i], nv2_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_nv2_z[i], nv2_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);

        cudaMemcpy(d_nv3_x[i], nv3_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_nv3_y[i], nv3_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_nv3_z[i], nv3_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);

        cudaMemcpy(d_nv1_prime_x[i], nv1_prime_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_nv1_prime_y[i], nv1_prime_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_nv1_prime_z[i], nv1_prime_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);

        cudaMemcpy(d_nv2_prime_x[i], nv2_prime_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_nv2_prime_y[i], nv2_prime_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_nv2_prime_z[i], nv2_prime_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);

        cudaMemcpy(d_nv3_prime_x[i], nv3_prime_x+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_nv3_prime_y[i], nv3_prime_y+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_nv3_prime_z[i], nv3_prime_z+start_idx, block_n*sizeof(double), cudaMemcpyHostToDevice);

        cudaMemcpy(d_times[i], times+start_idx, block_n*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_still_together[i], still_together+start_idx, block_n*sizeof(bool), cudaMemcpyHostToDevice);
        cudaMemcpy(d_not_diverged[i], not_diverged+start_idx, block_n*sizeof(bool), cudaMemcpyHostToDevice);

        // call CUDA kernal on inputs in configuration <<< blockIdx, threadIdx>>>>
        divergence<<<(block_n+127)/128, 128>>>(
            block_n, 
            steps, 
            delta_t,
            d_still_together[i],
            d_not_diverged[i],
            d_times[i],
            m1, m2, m3,
            critical_distance,
            d_p1_x[i], d_p1_y[i], d_p1_z[i], 
            d_p2_x[i], d_p2_y[i], d_p2_z[i], 
            d_p3_x[i], d_p3_y[i], d_p3_z[i], 
            d_p1_prime_x[i], d_p1_prime_y[i], d_p1_prime_z[i], 
            d_p2_prime_x[i], d_p2_prime_y[i], d_p2_prime_z[i], 
            d_p3_prime_x[i], d_p3_prime_y[i], d_p3_prime_z[i],
            d_dv_1_x[i], d_dv_1_y[i], d_dv_1_z[i],
            d_dv_2_x[i], d_dv_2_y[i], d_dv_2_z[i],
            d_dv_3_x[i], d_dv_3_y[i], d_dv_3_z[i],
            d_dv_1pr_x[i], d_dv_1pr_y[i], d_dv_1pr_z[i],
            d_dv_2pr_x[i], d_dv_2pr_y[i], d_dv_2pr_z[i],
            d_dv_3pr_x[i], d_dv_3pr_y[i], d_dv_3pr_z[i],
            d_v1_x[i], d_v1_y[i], d_v1_z[i],
            d_v2_x[i], d_v2_y[i], d_v2_z[i],
            d_v3_x[i], d_v3_y[i], d_v3_z[i],
            d_v1_prime_x[i], d_v1_prime_y[i], d_v1_prime_z[i],
            d_v2_prime_x[i], d_v2_prime_y[i], d_v2_prime_z[i],
            d_v3_prime_x[i], d_v3_prime_y[i], d_v3_prime_z[i],
            d_nv1_x[i], d_nv1_y[i], d_nv1_z[i],
            d_nv2_x[i], d_nv2_y[i], d_nv2_z[i],
            d_nv3_x[i], d_nv3_y[i], d_nv3_z[i],
            d_nv1_prime_x[i], d_nv1_prime_y[i], d_nv1_prime_z[i],
            d_nv2_prime_x[i], d_nv2_prime_y[i], d_nv2_prime_z[i],
            d_nv3_prime_x[i], d_nv3_prime_y[i], d_nv3_prime_z[i]
            );

        cudaMemcpyAsync(times+start_idx, d_times[i], block_n*sizeof(int), cudaMemcpyDeviceToHost);
        // cudaMemcpyAsync(still_together+start_idx, d_still_together, block_n*sizeof(bool), cudaMemcpyDeviceToHost);
        // cudaMemcpyAsync(not_diverged+start_idx, d_not_diverged, block_n*sizeof(bool), cudaMemcpyDeviceToHost);
        // cudaMemcpyAsync(p1_x+start_idx, d_p1_x[i], block_n*sizeof(double), cudaMemcpyDeviceToHost);
        // cudaMemcpyAsync(p1_y+start_idx, d_p1_y, block_n*sizeof(double), cudaMemcpyDeviceToHost);
        // cudaMemcpyAsync(p1_z+start_idx, d_p1_z, block_n*sizeof(double), cudaMemcpyDeviceToHost);
        // cudaMemcpyAsync(p1_prime_x+start_idx, d_p1_prime_x, block_n*sizeof(double), cudaMemcpyDeviceToHost);
        // cudaMemcpyAsync(p1_prime_y+start_idx, d_p1_prime_y, block_n*sizeof(double), cudaMemcpyDeviceToHost);
        // cudaMemcpyAsync(p1_prime_z+start_idx, d_p1_prime_z, block_n*sizeof(double), cudaMemcpyDeviceToHost);
    }

    cudaDeviceSynchronize();
    // measure elapsed kernal runtime
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Elapsed Time: " << elapsed_seconds.count() << "s\n";

    cudaMemcpy(times, d_times, N*sizeof(int), cudaMemcpyDeviceToHost);
    for (int i=0; i<n_gpus; i++){
      std::cout << "GPU number " << i << " deallocating" << "\n";
      cudaSetDevice(i);
      std::cout << &d_p1_x[i];
      cudaFree(d_p1_x[i]); cudaFree(d_p1_y[i]); cudaFree(d_p1_z[i]);
      cudaFree(d_p2_x[i]); cudaFree(d_p2_y[i]); cudaFree(d_p2_z[i]);
      cudaFree(d_p3_x[i]); cudaFree(d_p3_y[i]); cudaFree(d_p3_z[i]);

      cudaFree(d_p1_prime_x[i]); cudaFree(d_p1_prime_y[i]); cudaFree(d_p1_prime_z[i]);  
      cudaFree(d_p2_prime_x[i]); cudaFree(d_p2_prime_y[i]); cudaFree(d_p2_prime_z[i]);
      cudaFree(d_p3_prime_x[i]); cudaFree(d_p3_prime_y[i]); cudaFree(d_p3_prime_z[i]);

      cudaFree(d_dv_1_x[i]); cudaFree(d_dv_1_y[i]); cudaFree(d_dv_1_z[i]);
      cudaFree(d_dv_2_x[i]); cudaFree(d_dv_2_y[i]); cudaFree(d_dv_2_z[i]);
      cudaFree(d_dv_3_x[i]); cudaFree(d_dv_3_y[i]); cudaFree(d_dv_3_z[i]);

      cudaFree(d_dv_1pr_x[i]); cudaFree(d_dv_1pr_y[i]); cudaFree(d_dv_1pr_z[i]);
      cudaFree(d_dv_2pr_x[i]); cudaFree(d_dv_2pr_y[i]); cudaFree(d_dv_2pr_z[i]);
      cudaFree(d_dv_3pr_x[i]); cudaFree(d_dv_3pr_y[i]); cudaFree(d_dv_3pr_z[i]);

      cudaFree(d_nv1_x[i]); cudaFree(d_nv1_y[i]); cudaFree(d_nv1_z[i]);
      cudaFree(d_nv2_x[i]); cudaFree(d_nv2_y[i]); cudaFree(d_nv2_z[i]);
      cudaFree(d_nv3_x[i]); cudaFree(d_nv3_y[i]); cudaFree(d_nv3_z[i]);

      cudaFree(d_v1_x[i]); cudaFree(d_v1_y[i]); cudaFree(d_v1_z[i]);
      cudaFree(d_v2_x[i]); cudaFree(d_v2_y[i]); cudaFree(d_v2_z[i]);
      cudaFree(d_v3_x[i]); cudaFree(d_v3_y[i]); cudaFree(d_v3_z[i]);

      cudaFree(d_v1_prime_x[i]); cudaFree(d_v1_prime_y[i]); cudaFree(d_v1_prime_z[i]);
      cudaFree(d_v2_prime_x[i]); cudaFree(d_v2_prime_y[i]); cudaFree(d_v2_prime_z[i]);
      cudaFree(d_v3_prime_x[i]); cudaFree(d_v3_prime_y[i]); cudaFree(d_v3_prime_z[i]);

      cudaFree(d_nv1_prime_x[i]); cudaFree(d_nv1_prime_y[i]); cudaFree(d_nv1_prime_z[i]);
      cudaFree(d_nv2_prime_x[i]); cudaFree(d_nv2_prime_y[i]); cudaFree(d_nv2_prime_z[i]);
      cudaFree(d_nv3_prime_x[i]); cudaFree(d_nv3_prime_y[i]); cudaFree(d_nv3_prime_z[i]);

      cudaFree(d_times[i]); cudaFree(d_still_together[i]); cudaFree(d_not_diverged[i]);
    }

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
    return times;
    }
  }














