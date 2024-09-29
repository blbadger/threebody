#include <stdio.h>
#include <iostream>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <distributed.h>

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