#include <stdio.h>
#include <iostream>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define SHIFT_AMOUNT 10
#define SHIFT_MASK ((1 << SHIFT_AMOUNT) - 1) 


// kernal declaration
__global__
void divergence(int n, 
              int steps,
              int delta_t,
              bool *still_together,
              bool *not_diverged,
              int *times,
              int m_1, int m_2, int m_3,
              int critical_distance,
              int *p1_x, int *p1_y, int *p1_z, 
              int *p2_x, int *p2_y, int *p2_z, 
              int *p3_x, int *p3_y, int *p3_z, 
              int *p1_prime_x, int *p1_prime_y, int *p1_prime_z, 
              int *p2_prime_x, int *p2_prime_y, int *p2_prime_z, 
              int *p3_prime_x, int *p3_prime_y, int *p3_prime_z, 
              int *dv_1_x, int *dv_1_y, int *dv_1_z,
              int *dv_2_x, int *dv_2_y, int *dv_2_z,
              int *dv_3_x, int *dv_3_y, int *dv_3_z,
              int *dv_1pr_x, int *dv_1pr_y, int *dv_1pr_z,
              int *dv_2pr_x, int *dv_2pr_y, int *dv_2pr_z,
              int *dv_3pr_x, int *dv_3pr_y, int *dv_3pr_z,
              int *v1_x, int *v1_y, int *v1_z,
              int *v2_x, int *v2_y, int *v2_z,
              int *v3_x, int *v3_y, int *v3_z,
              int *v1_prime_x, int *v1_prime_y, int *v1_prime_z,
              int *v2_prime_x, int *v2_prime_y, int *v2_prime_z,
              int *v3_prime_x, int *v3_prime_y, int *v3_prime_z,
              int *nv1_x, int *nv1_y, int *nv1_z,
              int *nv2_x, int *nv2_y, int *nv2_z,
              int *nv3_x, int *nv3_y, int *nv3_z,
              int *nv1_prime_x, int *nv1_prime_y, int *nv1_prime_z,
              int *nv2_prime_x, int *nv2_prime_y, int *nv2_prime_z,
              int *nv3_prime_x, int *nv3_prime_y, int *nv3_prime_z
              )
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int g = -(98 << (SHIFT_AMOUNT))/10;
  for (int j=0; j < steps; j++) {
    if (i < n){
      // compute accelerations
      dv_1_x[i] = g * m_2 * (p1_x[i] - p2_x[i]) / (sqrt((p1_x[i] - p2_x[i])*(p1_x[i] - p2_x[i]) + (p1_y[i] - p2_y[i])*(p1_y[i] - p2_y[i]) + (p1_z[i] - p2_z[i])*(p1_z[i] - p2_z[i]))*sqrt((p1_x[i] - p2_x[i])*(p1_x[i] - p2_x[i]) + (p1_y[i] - p2_y[i])*(p1_y[i] - p2_y[i]) + (p1_z[i] - p2_z[i])*(p1_z[i] - p2_z[i]))*sqrt((p1_x[i] - p2_x[i])*(p1_x[i] - p2_x[i]) + (p1_y[i] - p2_y[i])*(p1_y[i] - p2_y[i]) + (p1_z[i] - p2_z[i])*(p1_z[i] - p2_z[i]))) -9.8 * m_3 * (p1_x[i] - p3_x[i]) / (sqrt((p1_x[i] - p3_x[i])*(p1_x[i] - p3_x[i]) + (p1_y[i] - p3_y[i])*(p1_y[i] - p3_y[i]) + (p1_z[i] - p3_z[i])*(p1_z[i] - p3_z[i]))*sqrt((p1_x[i] - p3_x[i])*(p1_x[i] - p3_x[i]) + (p1_y[i] - p3_y[i])*(p1_y[i] - p3_y[i]) + (p1_z[i] - p3_z[i])*(p1_z[i] - p3_z[i]))*sqrt((p1_x[i] - p3_x[i])*(p1_x[i] - p3_x[i]) + (p1_y[i] - p3_y[i])*(p1_y[i] - p3_y[i]) + (p1_z[i] - p3_z[i])*(p1_z[i] - p3_z[i])));
      dv_1_y[i] = g * m_2 * (p1_y[i] - p2_y[i]) / (sqrt((p1_x[i] - p2_x[i])*(p1_x[i] - p2_x[i]) + (p1_y[i] - p2_y[i])*(p1_y[i] - p2_y[i]) + (p1_z[i] - p2_z[i])*(p1_z[i] - p2_z[i]))*sqrt((p1_x[i] - p2_x[i])*(p1_x[i] - p2_x[i]) + (p1_y[i] - p2_y[i])*(p1_y[i] - p2_y[i]) + (p1_z[i] - p2_z[i])*(p1_z[i] - p2_z[i]))*sqrt((p1_x[i] - p2_x[i])*(p1_x[i] - p2_x[i]) + (p1_y[i] - p2_y[i])*(p1_y[i] - p2_y[i]) + (p1_z[i] - p2_z[i])*(p1_z[i] - p2_z[i]))) -9.8 * m_3 * (p1_y[i] - p3_y[i]) / (sqrt((p1_x[i] - p3_x[i])*(p1_x[i] - p3_x[i]) + (p1_y[i] - p3_y[i])*(p1_y[i] - p3_y[i]) + (p1_z[i] - p3_z[i])*(p1_z[i] - p3_z[i]))*sqrt((p1_x[i] - p3_x[i])*(p1_x[i] - p3_x[i]) + (p1_y[i] - p3_y[i])*(p1_y[i] - p3_y[i]) + (p1_z[i] - p3_z[i])*(p1_z[i] - p3_z[i]))*sqrt((p1_x[i] - p3_x[i])*(p1_x[i] - p3_x[i]) + (p1_y[i] - p3_y[i])*(p1_y[i] - p3_y[i]) + (p1_z[i] - p3_z[i])*(p1_z[i] - p3_z[i])));
      dv_1_z[i] = g* m_2 * (p1_z[i] - p2_z[i]) / (sqrt((p1_x[i] - p2_x[i])*(p1_x[i] - p2_x[i]) + (p1_y[i] - p2_y[i])*(p1_y[i] - p2_y[i]) + (p1_z[i] - p2_z[i])*(p1_z[i] - p2_z[i]))*sqrt((p1_x[i] - p2_x[i])*(p1_x[i] - p2_x[i]) + (p1_y[i] - p2_y[i])*(p1_y[i] - p2_y[i]) + (p1_z[i] - p2_z[i])*(p1_z[i] - p2_z[i]))*sqrt((p1_x[i] - p2_x[i])*(p1_x[i] - p2_x[i]) + (p1_y[i] - p2_y[i])*(p1_y[i] - p2_y[i]) + (p1_z[i] - p2_z[i])*(p1_z[i] - p2_z[i]))) -9.8 * m_3 * (p1_z[i] - p3_z[i]) / (sqrt((p1_x[i] - p3_x[i])*(p1_x[i] - p3_x[i]) + (p1_y[i] - p3_y[i])*(p1_y[i] - p3_y[i]) + (p1_z[i] - p3_z[i])*(p1_z[i] - p3_z[i]))*sqrt((p1_x[i] - p3_x[i])*(p1_x[i] - p3_x[i]) + (p1_y[i] - p3_y[i])*(p1_y[i] - p3_y[i]) + (p1_z[i] - p3_z[i])*(p1_z[i] - p3_z[i]))*sqrt((p1_x[i] - p3_x[i])*(p1_x[i] - p3_x[i]) + (p1_y[i] - p3_y[i])*(p1_y[i] - p3_y[i]) + (p1_z[i] - p3_z[i])*(p1_z[i] - p3_z[i])));
      
      dv_2_x[i] = g * m_3 * (p2_x[i] - p3_x[i]) / (sqrt((p2_x[i] - p3_x[i])*(p2_x[i] - p3_x[i]) + (p2_y[i] - p3_y[i])*(p2_y[i] - p3_y[i]) + (p2_z[i] - p3_z[i])*(p2_z[i] - p3_z[i]))*sqrt((p2_x[i] - p3_x[i])*(p2_x[i] - p3_x[i]) + (p2_y[i] - p3_y[i])*(p2_y[i] - p3_y[i]) + (p2_z[i] - p3_z[i])*(p2_z[i] - p3_z[i]))*sqrt((p2_x[i] - p3_x[i])*(p2_x[i] - p3_x[i]) + (p2_y[i] - p3_y[i])*(p2_y[i] - p3_y[i]) + (p2_z[i] - p3_z[i])*(p2_z[i] - p3_z[i]))) -9.8 * m_1 * (p2_x[i] - p1_x[i]) / (sqrt((p2_x[i] - p1_x[i])*(p2_x[i] - p1_x[i]) + (p2_y[i] - p1_y[i])*(p2_y[i] - p1_y[i]) + (p2_z[i] - p1_z[i])*(p2_z[i] - p1_z[i]))*sqrt((p2_x[i] - p1_x[i])*(p2_x[i] - p1_x[i]) + (p2_y[i] - p1_y[i])*(p2_y[i] - p1_y[i]) + (p2_z[i] - p1_z[i])*(p2_z[i] - p1_z[i]))*sqrt((p2_x[i] - p1_x[i])*(p2_x[i] - p1_x[i]) + (p2_y[i] - p1_y[i])*(p2_y[i] - p1_y[i]) + (p2_z[i] - p1_z[i])*(p2_z[i] - p1_z[i])));
      dv_2_y[i] = g * m_3 * (p2_y[i] - p3_y[i]) / (sqrt((p2_x[i] - p3_x[i])*(p2_x[i] - p3_x[i]) + (p2_y[i] - p3_y[i])*(p2_y[i] - p3_y[i]) + (p2_z[i] - p3_z[i])*(p2_z[i] - p3_z[i]))*sqrt((p2_x[i] - p3_x[i])*(p2_x[i] - p3_x[i]) + (p2_y[i] - p3_y[i])*(p2_y[i] - p3_y[i]) + (p2_z[i] - p3_z[i])*(p2_z[i] - p3_z[i]))*sqrt((p2_x[i] - p3_x[i])*(p2_x[i] - p3_x[i]) + (p2_y[i] - p3_y[i])*(p2_y[i] - p3_y[i]) + (p2_z[i] - p3_z[i])*(p2_z[i] - p3_z[i]))) -9.8 * m_1 * (p2_y[i] - p1_y[i]) / (sqrt((p2_x[i] - p1_x[i])*(p2_x[i] - p1_x[i]) + (p2_y[i] - p1_y[i])*(p2_y[i] - p1_y[i]) + (p2_z[i] - p1_z[i])*(p2_z[i] - p1_z[i]))*sqrt((p2_x[i] - p1_x[i])*(p2_x[i] - p1_x[i]) + (p2_y[i] - p1_y[i])*(p2_y[i] - p1_y[i]) + (p2_z[i] - p1_z[i])*(p2_z[i] - p1_z[i]))*sqrt((p2_x[i] - p1_x[i])*(p2_x[i] - p1_x[i]) + (p2_y[i] - p1_y[i])*(p2_y[i] - p1_y[i]) + (p2_z[i] - p1_z[i])*(p2_z[i] - p1_z[i])));
      dv_2_z[i] = g * m_3 * (p2_z[i] - p3_z[i]) / (sqrt((p2_x[i] - p3_x[i])*(p2_x[i] - p3_x[i]) + (p2_y[i] - p3_y[i])*(p2_y[i] - p3_y[i]) + (p2_z[i] - p3_z[i])*(p2_z[i] - p3_z[i]))*sqrt((p2_x[i] - p3_x[i])*(p2_x[i] - p3_x[i]) + (p2_y[i] - p3_y[i])*(p2_y[i] - p3_y[i]) + (p2_z[i] - p3_z[i])*(p2_z[i] - p3_z[i]))*sqrt((p2_x[i] - p3_x[i])*(p2_x[i] - p3_x[i]) + (p2_y[i] - p3_y[i])*(p2_y[i] - p3_y[i]) + (p2_z[i] - p3_z[i])*(p2_z[i] - p3_z[i]))) -9.8 * m_1 * (p2_z[i] - p1_z[i]) / (sqrt((p2_x[i] - p1_x[i])*(p2_x[i] - p1_x[i]) + (p2_y[i] - p1_y[i])*(p2_y[i] - p1_y[i]) + (p2_z[i] - p1_z[i])*(p2_z[i] - p1_z[i]))*sqrt((p2_x[i] - p1_x[i])*(p2_x[i] - p1_x[i]) + (p2_y[i] - p1_y[i])*(p2_y[i] - p1_y[i]) + (p2_z[i] - p1_z[i])*(p2_z[i] - p1_z[i]))*sqrt((p2_x[i] - p1_x[i])*(p2_x[i] - p1_x[i]) + (p2_y[i] - p1_y[i])*(p2_y[i] - p1_y[i]) + (p2_z[i] - p1_z[i])*(p2_z[i] - p1_z[i])));

      dv_3_x[i] = -(98 << (SHIFT_AMOUNT))/10 * m_1 * (p3_x[i] - p1_x[i]) / (sqrt((p3_x[i] - p1_x[i])*(p3_x[i] - p1_x[i]) + (p3_y[i] - p1_y[i])*(p3_y[i] - p1_y[i]) + (p3_z[i] - p1_z[i])*(p3_z[i] - p1_z[i]))*sqrt((p3_x[i] - p1_x[i])*(p3_x[i] - p1_x[i]) + (p3_y[i] - p1_y[i])*(p3_y[i] - p1_y[i]) + (p3_z[i] - p1_z[i])*(p3_z[i] - p1_z[i]))*sqrt((p3_x[i] - p1_x[i])*(p3_x[i] - p1_x[i]) + (p3_y[i] - p1_y[i])*(p3_y[i] - p1_y[i]) + (p3_z[i] - p1_z[i])*(p3_z[i] - p1_z[i]))) -9.8 * m_2 * (p3_x[i] - p2_x[i]) / (sqrt((p3_x[i] - p2_x[i])*(p3_x[i] - p2_x[i]) + (p3_y[i] - p2_y[i])*(p3_y[i] - p2_y[i]) + (p3_z[i] - p2_z[i])*(p3_z[i] - p2_z[i]))*sqrt((p3_x[i] - p2_x[i])*(p3_x[i] - p2_x[i]) + (p3_y[i] - p2_y[i])*(p3_y[i] - p2_y[i]) + (p3_z[i] - p2_z[i])*(p3_z[i] - p2_z[i]))*sqrt((p3_x[i] - p2_x[i])*(p3_x[i] - p2_x[i]) + (p3_y[i] - p2_y[i])*(p3_y[i] - p2_y[i]) + (p3_z[i] - p2_z[i])*(p3_z[i] - p2_z[i])));
      dv_3_y[i] = -(98 << (SHIFT_AMOUNT))/10 * m_1 * (p3_y[i] - p1_y[i]) / (sqrt((p3_x[i] - p1_x[i])*(p3_x[i] - p1_x[i]) + (p3_y[i] - p1_y[i])*(p3_y[i] - p1_y[i]) + (p3_z[i] - p1_z[i])*(p3_z[i] - p1_z[i]))*sqrt((p3_x[i] - p1_x[i])*(p3_x[i] - p1_x[i]) + (p3_y[i] - p1_y[i])*(p3_y[i] - p1_y[i]) + (p3_z[i] - p1_z[i])*(p3_z[i] - p1_z[i]))*sqrt((p3_x[i] - p1_x[i])*(p3_x[i] - p1_x[i]) + (p3_y[i] - p1_y[i])*(p3_y[i] - p1_y[i]) + (p3_z[i] - p1_z[i])*(p3_z[i] - p1_z[i]))) -9.8 * m_2 * (p3_y[i] - p2_y[i]) / (sqrt((p3_x[i] - p2_x[i])*(p3_x[i] - p2_x[i]) + (p3_y[i] - p2_y[i])*(p3_y[i] - p2_y[i]) + (p3_z[i] - p2_z[i])*(p3_z[i] - p2_z[i]))*sqrt((p3_x[i] - p2_x[i])*(p3_x[i] - p2_x[i]) + (p3_y[i] - p2_y[i])*(p3_y[i] - p2_y[i]) + (p3_z[i] - p2_z[i])*(p3_z[i] - p2_z[i]))*sqrt((p3_x[i] - p2_x[i])*(p3_x[i] - p2_x[i]) + (p3_y[i] - p2_y[i])*(p3_y[i] - p2_y[i]) + (p3_z[i] - p2_z[i])*(p3_z[i] - p2_z[i])));
      dv_3_z[i] = -(98 << (SHIFT_AMOUNT))/10 * m_1 * (p3_z[i] - p1_z[i]) / (sqrt((p3_x[i] - p1_x[i])*(p3_x[i] - p1_x[i]) + (p3_y[i] - p1_y[i])*(p3_y[i] - p1_y[i]) + (p3_z[i] - p1_z[i])*(p3_z[i] - p1_z[i]))*sqrt((p3_x[i] - p1_x[i])*(p3_x[i] - p1_x[i]) + (p3_y[i] - p1_y[i])*(p3_y[i] - p1_y[i]) + (p3_z[i] - p1_z[i])*(p3_z[i] - p1_z[i]))*sqrt((p3_x[i] - p1_x[i])*(p3_x[i] - p1_x[i]) + (p3_y[i] - p1_y[i])*(p3_y[i] - p1_y[i]) + (p3_z[i] - p1_z[i])*(p3_z[i] - p1_z[i]))) -9.8 * m_2 * (p3_z[i] - p2_z[i]) / (sqrt((p3_x[i] - p2_x[i])*(p3_x[i] - p2_x[i]) + (p3_y[i] - p2_y[i])*(p3_y[i] - p2_y[i]) + (p3_z[i] - p2_z[i])*(p3_z[i] - p2_z[i]))*sqrt((p3_x[i] - p2_x[i])*(p3_x[i] - p2_x[i]) + (p3_y[i] - p2_y[i])*(p3_y[i] - p2_y[i]) + (p3_z[i] - p2_z[i])*(p3_z[i] - p2_z[i]))*sqrt((p3_x[i] - p2_x[i])*(p3_x[i] - p2_x[i]) + (p3_y[i] - p2_y[i])*(p3_y[i] - p2_y[i]) + (p3_z[i] - p2_z[i])*(p3_z[i] - p2_z[i])));

      dv_1pr_x[i] = -(98 << (SHIFT_AMOUNT))/10 * m_2 * (p1_prime_x[i] - p2_prime_x[i]) / (sqrt((p1_prime_x[i] - p2_prime_x[i])*(p1_prime_x[i] - p2_prime_x[i]) + (p1_prime_y[i] - p2_prime_y[i])*(p1_prime_y[i] - p2_prime_y[i]) + (p1_prime_z[i] - p2_prime_z[i])*(p1_prime_z[i] - p2_prime_z[i]))*sqrt((p1_prime_x[i] - p2_prime_x[i])*(p1_prime_x[i] - p2_prime_x[i]) + (p1_prime_y[i] - p2_prime_y[i])*(p1_prime_y[i] - p2_prime_y[i]) + (p1_prime_z[i] - p2_prime_z[i])*(p1_prime_z[i] - p2_prime_z[i]))*sqrt((p1_prime_x[i] - p2_prime_x[i])*(p1_prime_x[i] - p2_prime_x[i]) + (p1_prime_y[i] - p2_prime_y[i])*(p1_prime_y[i] - p2_prime_y[i]) + (p1_prime_z[i] - p2_prime_z[i])*(p1_prime_z[i] - p2_prime_z[i]))) -9.8 * m_3 * (p1_prime_x[i] - p3_prime_x[i]) / (sqrt((p1_prime_x[i] - p3_prime_x[i])*(p1_prime_x[i] - p3_prime_x[i]) + (p1_prime_y[i] - p3_prime_y[i])*(p1_prime_y[i] - p3_prime_y[i]) + (p1_prime_z[i] - p3_prime_z[i])*(p1_prime_z[i] - p3_prime_z[i]))*sqrt((p1_prime_x[i] - p3_prime_x[i])*(p1_prime_x[i] - p3_prime_x[i]) + (p1_prime_y[i] - p3_prime_y[i])*(p1_prime_y[i] - p3_prime_y[i]) + (p1_prime_z[i] - p3_prime_z[i])*(p1_prime_z[i] - p3_prime_z[i]))*sqrt((p1_prime_x[i] - p3_prime_x[i])*(p1_prime_x[i] - p3_prime_x[i]) + (p1_prime_y[i] - p3_prime_y[i])*(p1_prime_y[i] - p3_prime_y[i]) + (p1_prime_z[i] - p3_prime_z[i])*(p1_prime_z[i] - p3_prime_z[i])));
      dv_1pr_y[i] = -(98 << (SHIFT_AMOUNT))/10 * m_2 * (p1_prime_y[i] - p2_prime_y[i]) / (sqrt((p1_prime_x[i] - p2_prime_x[i])*(p1_prime_x[i] - p2_prime_x[i]) + (p1_prime_y[i] - p2_prime_y[i])*(p1_prime_y[i] - p2_prime_y[i]) + (p1_prime_z[i] - p2_prime_z[i])*(p1_prime_z[i] - p2_prime_z[i]))*sqrt((p1_prime_x[i] - p2_prime_x[i])*(p1_prime_x[i] - p2_prime_x[i]) + (p1_prime_y[i] - p2_prime_y[i])*(p1_prime_y[i] - p2_prime_y[i]) + (p1_prime_z[i] - p2_prime_z[i])*(p1_prime_z[i] - p2_prime_z[i]))*sqrt((p1_prime_x[i] - p2_prime_x[i])*(p1_prime_x[i] - p2_prime_x[i]) + (p1_prime_y[i] - p2_prime_y[i])*(p1_prime_y[i] - p2_prime_y[i]) + (p1_prime_z[i] - p2_prime_z[i])*(p1_prime_z[i] - p2_prime_z[i]))) -9.8 * m_3 * (p1_prime_y[i] - p3_prime_y[i]) / (sqrt((p1_prime_x[i] - p3_prime_x[i])*(p1_prime_x[i] - p3_prime_x[i]) + (p1_prime_y[i] - p3_prime_y[i])*(p1_prime_y[i] - p3_prime_y[i]) + (p1_prime_z[i] - p3_prime_z[i])*(p1_prime_z[i] - p3_prime_z[i]))*sqrt((p1_prime_x[i] - p3_prime_x[i])*(p1_prime_x[i] - p3_prime_x[i]) + (p1_prime_y[i] - p3_prime_y[i])*(p1_prime_y[i] - p3_prime_y[i]) + (p1_prime_z[i] - p3_prime_z[i])*(p1_prime_z[i] - p3_prime_z[i]))*sqrt((p1_prime_x[i] - p3_prime_x[i])*(p1_prime_x[i] - p3_prime_x[i]) + (p1_prime_y[i] - p3_prime_y[i])*(p1_prime_y[i] - p3_prime_y[i]) + (p1_prime_z[i] - p3_prime_z[i])*(p1_prime_z[i] - p3_prime_z[i])));
      dv_1pr_z[i] = -(98 << (SHIFT_AMOUNT))/10 * m_2 * (p1_prime_z[i] - p2_prime_z[i]) / (sqrt((p1_prime_x[i] - p2_prime_x[i])*(p1_prime_x[i] - p2_prime_x[i]) + (p1_prime_y[i] - p2_prime_y[i])*(p1_prime_y[i] - p2_prime_y[i]) + (p1_prime_z[i] - p2_prime_z[i])*(p1_prime_z[i] - p2_prime_z[i]))*sqrt((p1_prime_x[i] - p2_prime_x[i])*(p1_prime_x[i] - p2_prime_x[i]) + (p1_prime_y[i] - p2_prime_y[i])*(p1_prime_y[i] - p2_prime_y[i]) + (p1_prime_z[i] - p2_prime_z[i])*(p1_prime_z[i] - p2_prime_z[i]))*sqrt((p1_prime_x[i] - p2_prime_x[i])*(p1_prime_x[i] - p2_prime_x[i]) + (p1_prime_y[i] - p2_prime_y[i])*(p1_prime_y[i] - p2_prime_y[i]) + (p1_prime_z[i] - p2_prime_z[i])*(p1_prime_z[i] - p2_prime_z[i]))) -9.8 * m_3 * (p1_prime_z[i] - p3_prime_z[i]) / (sqrt((p1_prime_x[i] - p3_prime_x[i])*(p1_prime_x[i] - p3_prime_x[i]) + (p1_prime_y[i] - p3_prime_y[i])*(p1_prime_y[i] - p3_prime_y[i]) + (p1_prime_z[i] - p3_prime_z[i])*(p1_prime_z[i] - p3_prime_z[i]))*sqrt((p1_prime_x[i] - p3_prime_x[i])*(p1_prime_x[i] - p3_prime_x[i]) + (p1_prime_y[i] - p3_prime_y[i])*(p1_prime_y[i] - p3_prime_y[i]) + (p1_prime_z[i] - p3_prime_z[i])*(p1_prime_z[i] - p3_prime_z[i]))*sqrt((p1_prime_x[i] - p3_prime_x[i])*(p1_prime_x[i] - p3_prime_x[i]) + (p1_prime_y[i] - p3_prime_y[i])*(p1_prime_y[i] - p3_prime_y[i]) + (p1_prime_z[i] - p3_prime_z[i])*(p1_prime_z[i] - p3_prime_z[i])));
      
      dv_2pr_x[i] = -(98 << (SHIFT_AMOUNT))/10 * m_3 * (p2_prime_x[i] - p3_prime_x[i]) / (sqrt((p2_prime_x[i] - p3_prime_x[i])*(p2_prime_x[i] - p3_prime_x[i]) + (p2_prime_y[i] - p3_prime_y[i])*(p2_prime_y[i] - p3_prime_y[i]) + (p2_prime_z[i] - p3_prime_z[i])*(p2_prime_z[i] - p3_prime_z[i]))*sqrt((p2_prime_x[i] - p3_prime_x[i])*(p2_prime_x[i] - p3_prime_x[i]) + (p2_prime_y[i] - p3_prime_y[i])*(p2_prime_y[i] - p3_prime_y[i]) + (p2_prime_z[i] - p3_prime_z[i])*(p2_prime_z[i] - p3_prime_z[i]))*sqrt((p2_prime_x[i] - p3_prime_x[i])*(p2_prime_x[i] - p3_prime_x[i]) + (p2_prime_y[i] - p3_prime_y[i])*(p2_prime_y[i] - p3_prime_y[i]) + (p2_prime_z[i] - p3_prime_z[i])*(p2_prime_z[i] - p3_prime_z[i]))) -9.8 * m_1 * (p2_prime_x[i] - p1_prime_x[i]) / (sqrt((p2_prime_x[i] - p1_prime_x[i])*(p2_prime_x[i] - p1_prime_x[i]) + (p2_prime_y[i] - p1_prime_y[i])*(p2_prime_y[i] - p1_prime_y[i]) + (p2_prime_z[i] - p1_prime_z[i])*(p2_prime_z[i] - p1_prime_z[i]))*sqrt((p2_prime_x[i] - p1_prime_x[i])*(p2_prime_x[i] - p1_prime_x[i]) + (p2_prime_y[i] - p1_prime_y[i])*(p2_prime_y[i] - p1_prime_y[i]) + (p2_prime_z[i] - p1_prime_z[i])*(p2_prime_z[i] - p1_prime_z[i]))*sqrt((p2_prime_x[i] - p1_prime_x[i])*(p2_prime_x[i] - p1_prime_x[i]) + (p2_prime_y[i] - p1_prime_y[i])*(p2_prime_y[i] - p1_prime_y[i]) + (p2_prime_z[i] - p1_prime_z[i])*(p2_prime_z[i] - p1_prime_z[i])));
      dv_2pr_y[i] = -(98 << (SHIFT_AMOUNT))/10 * m_3 * (p2_prime_y[i] - p3_prime_y[i]) / (sqrt((p2_prime_x[i] - p3_prime_x[i])*(p2_prime_x[i] - p3_prime_x[i]) + (p2_prime_y[i] - p3_prime_y[i])*(p2_prime_y[i] - p3_prime_y[i]) + (p2_prime_z[i] - p3_prime_z[i])*(p2_prime_z[i] - p3_prime_z[i]))*sqrt((p2_prime_x[i] - p3_prime_x[i])*(p2_prime_x[i] - p3_prime_x[i]) + (p2_prime_y[i] - p3_prime_y[i])*(p2_prime_y[i] - p3_prime_y[i]) + (p2_prime_z[i] - p3_prime_z[i])*(p2_prime_z[i] - p3_prime_z[i]))*sqrt((p2_prime_x[i] - p3_prime_x[i])*(p2_prime_x[i] - p3_prime_x[i]) + (p2_prime_y[i] - p3_prime_y[i])*(p2_prime_y[i] - p3_prime_y[i]) + (p2_prime_z[i] - p3_prime_z[i])*(p2_prime_z[i] - p3_prime_z[i]))) -9.8 * m_1 * (p2_prime_y[i] - p1_prime_y[i]) / (sqrt((p2_prime_x[i] - p1_prime_x[i])*(p2_prime_x[i] - p1_prime_x[i]) + (p2_prime_y[i] - p1_prime_y[i])*(p2_prime_y[i] - p1_prime_y[i]) + (p2_prime_z[i] - p1_prime_z[i])*(p2_prime_z[i] - p1_prime_z[i]))*sqrt((p2_prime_x[i] - p1_prime_x[i])*(p2_prime_x[i] - p1_prime_x[i]) + (p2_prime_y[i] - p1_prime_y[i])*(p2_prime_y[i] - p1_prime_y[i]) + (p2_prime_z[i] - p1_prime_z[i])*(p2_prime_z[i] - p1_prime_z[i]))*sqrt((p2_prime_x[i] - p1_prime_x[i])*(p2_prime_x[i] - p1_prime_x[i]) + (p2_prime_y[i] - p1_prime_y[i])*(p2_prime_y[i] - p1_prime_y[i]) + (p2_prime_z[i] - p1_prime_z[i])*(p2_prime_z[i] - p1_prime_z[i])));
      dv_2pr_z[i] = -(98 << (SHIFT_AMOUNT))/10 * m_3 * (p2_prime_z[i] - p3_prime_z[i]) / (sqrt((p2_prime_x[i] - p3_prime_x[i])*(p2_prime_x[i] - p3_prime_x[i]) + (p2_prime_y[i] - p3_prime_y[i])*(p2_prime_y[i] - p3_prime_y[i]) + (p2_prime_z[i] - p3_prime_z[i])*(p2_prime_z[i] - p3_prime_z[i]))*sqrt((p2_prime_x[i] - p3_prime_x[i])*(p2_prime_x[i] - p3_prime_x[i]) + (p2_prime_y[i] - p3_prime_y[i])*(p2_prime_y[i] - p3_prime_y[i]) + (p2_prime_z[i] - p3_prime_z[i])*(p2_prime_z[i] - p3_prime_z[i]))*sqrt((p2_prime_x[i] - p3_prime_x[i])*(p2_prime_x[i] - p3_prime_x[i]) + (p2_prime_y[i] - p3_prime_y[i])*(p2_prime_y[i] - p3_prime_y[i]) + (p2_prime_z[i] - p3_prime_z[i])*(p2_prime_z[i] - p3_prime_z[i]))) -9.8 * m_1 * (p2_prime_z[i] - p1_prime_z[i]) / (sqrt((p2_prime_x[i] - p1_prime_x[i])*(p2_prime_x[i] - p1_prime_x[i]) + (p2_prime_y[i] - p1_prime_y[i])*(p2_prime_y[i] - p1_prime_y[i]) + (p2_prime_z[i] - p1_prime_z[i])*(p2_prime_z[i] - p1_prime_z[i]))*sqrt((p2_prime_x[i] - p1_prime_x[i])*(p2_prime_x[i] - p1_prime_x[i]) + (p2_prime_y[i] - p1_prime_y[i])*(p2_prime_y[i] - p1_prime_y[i]) + (p2_prime_z[i] - p1_prime_z[i])*(p2_prime_z[i] - p1_prime_z[i]))*sqrt((p2_prime_x[i] - p1_prime_x[i])*(p2_prime_x[i] - p1_prime_x[i]) + (p2_prime_y[i] - p1_prime_y[i])*(p2_prime_y[i] - p1_prime_y[i]) + (p2_prime_z[i] - p1_prime_z[i])*(p2_prime_z[i] - p1_prime_z[i])));

      dv_3pr_x[i] = -(98 << (SHIFT_AMOUNT))/10 * m_1 * (p3_prime_x[i] - p1_prime_x[i]) / (sqrt((p3_prime_x[i] - p1_prime_x[i])*(p3_prime_x[i] - p1_prime_x[i]) + (p3_prime_y[i] - p1_prime_y[i])*(p3_prime_y[i] - p1_prime_y[i]) + (p3_prime_z[i] - p1_prime_z[i])*(p3_prime_z[i] - p1_prime_z[i]))*sqrt((p3_prime_x[i] - p1_prime_x[i])*(p3_prime_x[i] - p1_prime_x[i]) + (p3_prime_y[i] - p1_prime_y[i])*(p3_prime_y[i] - p1_prime_y[i]) + (p3_prime_z[i] - p1_prime_z[i])*(p3_prime_z[i] - p1_prime_z[i]))*sqrt((p3_prime_x[i] - p1_prime_x[i])*(p3_prime_x[i] - p1_prime_x[i]) + (p3_prime_y[i] - p1_prime_y[i])*(p3_prime_y[i] - p1_prime_y[i]) + (p3_prime_z[i] - p1_prime_z[i])*(p3_prime_z[i] - p1_prime_z[i]))) -9.8 * m_2 * (p3_prime_x[i] - p2_prime_x[i]) / (sqrt((p3_prime_x[i] - p2_prime_x[i])*(p3_prime_x[i] - p2_prime_x[i]) + (p3_prime_y[i] - p2_prime_y[i])*(p3_prime_y[i] - p2_prime_y[i]) + (p3_prime_z[i] - p2_prime_z[i])*(p3_prime_z[i] - p2_prime_z[i]))*sqrt((p3_prime_x[i] - p2_prime_x[i])*(p3_prime_x[i] - p2_prime_x[i]) + (p3_prime_y[i] - p2_prime_y[i])*(p3_prime_y[i] - p2_prime_y[i]) + (p3_prime_z[i] - p2_prime_z[i])*(p3_prime_z[i] - p2_prime_z[i]))*sqrt((p3_prime_x[i] - p2_prime_x[i])*(p3_prime_x[i] - p2_prime_x[i]) + (p3_prime_y[i] - p2_prime_y[i])*(p3_prime_y[i] - p2_prime_y[i]) + (p3_prime_z[i] - p2_prime_z[i])*(p3_prime_z[i] - p2_prime_z[i])));
      dv_3pr_y[i] = -(98 << (SHIFT_AMOUNT))/10 * m_1 * (p3_prime_y[i] - p1_prime_y[i]) / (sqrt((p3_prime_x[i] - p1_prime_x[i])*(p3_prime_x[i] - p1_prime_x[i]) + (p3_prime_y[i] - p1_prime_y[i])*(p3_prime_y[i] - p1_prime_y[i]) + (p3_prime_z[i] - p1_prime_z[i])*(p3_prime_z[i] - p1_prime_z[i]))*sqrt((p3_prime_x[i] - p1_prime_x[i])*(p3_prime_x[i] - p1_prime_x[i]) + (p3_prime_y[i] - p1_prime_y[i])*(p3_prime_y[i] - p1_prime_y[i]) + (p3_prime_z[i] - p1_prime_z[i])*(p3_prime_z[i] - p1_prime_z[i]))*sqrt((p3_prime_x[i] - p1_prime_x[i])*(p3_prime_x[i] - p1_prime_x[i]) + (p3_prime_y[i] - p1_prime_y[i])*(p3_prime_y[i] - p1_prime_y[i]) + (p3_prime_z[i] - p1_prime_z[i])*(p3_prime_z[i] - p1_prime_z[i]))) -9.8 * m_2 * (p3_prime_y[i] - p2_prime_y[i]) / (sqrt((p3_prime_x[i] - p2_prime_x[i])*(p3_prime_x[i] - p2_prime_x[i]) + (p3_prime_y[i] - p2_prime_y[i])*(p3_prime_y[i] - p2_prime_y[i]) + (p3_prime_z[i] - p2_prime_z[i])*(p3_prime_z[i] - p2_prime_z[i]))*sqrt((p3_prime_x[i] - p2_prime_x[i])*(p3_prime_x[i] - p2_prime_x[i]) + (p3_prime_y[i] - p2_prime_y[i])*(p3_prime_y[i] - p2_prime_y[i]) + (p3_prime_z[i] - p2_prime_z[i])*(p3_prime_z[i] - p2_prime_z[i]))*sqrt((p3_prime_x[i] - p2_prime_x[i])*(p3_prime_x[i] - p2_prime_x[i]) + (p3_prime_y[i] - p2_prime_y[i])*(p3_prime_y[i] - p2_prime_y[i]) + (p3_prime_z[i] - p2_prime_z[i])*(p3_prime_z[i] - p2_prime_z[i])));
      dv_3pr_z[i] = -(98 << (SHIFT_AMOUNT))/10 * m_1 * (p3_prime_z[i] - p1_prime_z[i]) / (sqrt((p3_prime_x[i] - p1_prime_x[i])*(p3_prime_x[i] - p1_prime_x[i]) + (p3_prime_y[i] - p1_prime_y[i])*(p3_prime_y[i] - p1_prime_y[i]) + (p3_prime_z[i] - p1_prime_z[i])*(p3_prime_z[i] - p1_prime_z[i]))*sqrt((p3_prime_x[i] - p1_prime_x[i])*(p3_prime_x[i] - p1_prime_x[i]) + (p3_prime_y[i] - p1_prime_y[i])*(p3_prime_y[i] - p1_prime_y[i]) + (p3_prime_z[i] - p1_prime_z[i])*(p3_prime_z[i] - p1_prime_z[i]))*sqrt((p3_prime_x[i] - p1_prime_x[i])*(p3_prime_x[i] - p1_prime_x[i]) + (p3_prime_y[i] - p1_prime_y[i])*(p3_prime_y[i] - p1_prime_y[i]) + (p3_prime_z[i] - p1_prime_z[i])*(p3_prime_z[i] - p1_prime_z[i]))) -9.8 * m_2 * (p3_prime_z[i] - p2_prime_z[i]) / (sqrt((p3_prime_x[i] - p2_prime_x[i])*(p3_prime_x[i] - p2_prime_x[i]) + (p3_prime_y[i] - p2_prime_y[i])*(p3_prime_y[i] - p2_prime_y[i]) + (p3_prime_z[i] - p2_prime_z[i])*(p3_prime_z[i] - p2_prime_z[i]))*sqrt((p3_prime_x[i] - p2_prime_x[i])*(p3_prime_x[i] - p2_prime_x[i]) + (p3_prime_y[i] - p2_prime_y[i])*(p3_prime_y[i] - p2_prime_y[i]) + (p3_prime_z[i] - p2_prime_z[i])*(p3_prime_z[i] - p2_prime_z[i]))*sqrt((p3_prime_x[i] - p2_prime_x[i])*(p3_prime_x[i] - p2_prime_x[i]) + (p3_prime_y[i] - p2_prime_y[i])*(p3_prime_y[i] - p2_prime_y[i]) + (p3_prime_z[i] - p2_prime_z[i])*(p3_prime_z[i] - p2_prime_z[i])));


      // find which trajectories have diverged and increment *times
      not_diverged[i] = (p1_x[i]-p1_prime_x[i])*(p1_x[i]-p1_prime_x[i]) + (p1_y[i]-p1_prime_y[i])*(p1_y[i]-p1_prime_y[i]) + (p1_z[i]-p1_prime_z[i])*(p1_z[i]-p1_prime_z[i]) <= (critical_distance)*(critical_distance);
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

int main(void){
    int N = 1000000;
    int steps = 50000;
    int delta_t = (1 << SHIFT_AMOUNT) / 1000;
    int critical_distance = (5 << SHIFT_AMOUNT) / 10; // 0.5
    int m1 = 10 << SHIFT_AMOUNT;
    int m2 = 20 << SHIFT_AMOUNT;
    int m3 = 30 << SHIFT_AMOUNT;
    int *p1_x, *p1_y, *p1_z;
    int *p2_x, *p2_y, *p2_z;
    int *p3_x, *p3_y, *p3_z;
    int *p1_prime_x, *p1_prime_y, *p1_prime_z;
    int *p2_prime_x, *p2_prime_y, *p2_prime_z;
    int *p3_prime_x, *p3_prime_y, *p3_prime_z;
    int *dv_1_x, *dv_1_y, *dv_1_z;
    int *dv_2_x, *dv_2_y, *dv_2_z;
    int *dv_3_x, *dv_3_y, *dv_3_z;
    int *dv_1pr_x, *dv_1pr_y, *dv_1pr_z;
    int *dv_2pr_x, *dv_2pr_y, *dv_2pr_z;
    int *dv_3pr_x, *dv_3pr_y, *dv_3pr_z;
    int *v1_x, *v1_y, *v1_z;
    int *v2_x, *v2_y, *v2_z;
    int *v3_x, *v3_y, *v3_z;
    int *v1_prime_x, *v1_prime_y, *v1_prime_z;
    int *v2_prime_x, *v2_prime_y, *v2_prime_z;
    int *v3_prime_x, *v3_prime_y, *v3_prime_z;
    int *nv1_x, *nv1_y, *nv1_z;
    int *nv2_x, *nv2_y, *nv2_z;
    int *nv3_x, *nv3_y, *nv3_z;
    int *nv1_prime_x, *nv1_prime_y, *nv1_prime_z;
    int *nv2_prime_x, *nv2_prime_y, *nv2_prime_z;
    int *nv3_prime_x, *nv3_prime_y, *nv3_prime_z;

    int *d_p1_x, *d_p1_y, *d_p1_z;
    int *d_p2_x, *d_p2_y, *d_p2_z;
    int *d_p3_x, *d_p3_y, *d_p3_z;
    int *d_p1_prime_x, *d_p1_prime_y, *d_p1_prime_z;
    int *d_p2_prime_x, *d_p2_prime_y, *d_p2_prime_z;
    int *d_p3_prime_x, *d_p3_prime_y, *d_p3_prime_z;
    int *d_dv_1_x, *d_dv_1_y, *d_dv_1_z;
    int *d_dv_2_x, *d_dv_2_y, *d_dv_2_z;
    int *d_dv_3_x, *d_dv_3_y, *d_dv_3_z;
    int *d_dv_1pr_x, *d_dv_1pr_y, *d_dv_1pr_z;
    int *d_dv_2pr_x, *d_dv_2pr_y, *d_dv_2pr_z;
    int *d_dv_3pr_x, *d_dv_3pr_y, *d_dv_3pr_z;
    int *d_v1_x, *d_v1_y, *d_v1_z;
    int *d_v2_x, *d_v2_y, *d_v2_z;
    int *d_v3_x, *d_v3_y, *d_v3_z;
    int *d_v1_prime_x, *d_v1_prime_y, *d_v1_prime_z;
    int *d_v2_prime_x, *d_v2_prime_y, *d_v2_prime_z;
    int *d_v3_prime_x, *d_v3_prime_y, *d_v3_prime_z;
    int *d_nv1_x, *d_nv1_y, *d_nv1_z;
    int *d_nv2_x, *d_nv2_y, *d_nv2_z;
    int *d_nv3_x, *d_nv3_y, *d_nv3_z;
    int *d_nv1_prime_x, *d_nv1_prime_y, *d_nv1_prime_z;
    int *d_nv2_prime_x, *d_nv2_prime_y, *d_nv2_prime_z;
    int *d_nv3_prime_x, *d_nv3_prime_y, *d_nv3_prime_z;

    bool *still_together, *d_still_together;
    int *times, *d_times;
    bool *not_diverged, *d_not_diverged;

    p1_x = (int*)malloc(N*sizeof(int));
    p1_y = (int*)malloc(N*sizeof(int));
    p1_z = (int*)malloc(N*sizeof(int));

    p2_x = (int*)malloc(N*sizeof(int));
    p2_y = (int*)malloc(N*sizeof(int));
    p2_z = (int*)malloc(N*sizeof(int));

    p3_x = (int*)malloc(N*sizeof(int));
    p3_y = (int*)malloc(N*sizeof(int));
    p3_z = (int*)malloc(N*sizeof(int));

    p1_prime_x = (int*)malloc(N*sizeof(int));
    p1_prime_y = (int*)malloc(N*sizeof(int));
    p1_prime_z = (int*)malloc(N*sizeof(int));

    p2_prime_x = (int*)malloc(N*sizeof(int));
    p2_prime_y = (int*)malloc(N*sizeof(int));
    p2_prime_z = (int*)malloc(N*sizeof(int));

    p3_prime_x = (int*)malloc(N*sizeof(int));
    p3_prime_y = (int*)malloc(N*sizeof(int));
    p3_prime_z = (int*)malloc(N*sizeof(int));

    dv_1_x = (int*)malloc(N*sizeof(int));
    dv_1_y = (int*)malloc(N*sizeof(int));
    dv_1_z = (int*)malloc(N*sizeof(int));

    dv_2_x = (int*)malloc(N*sizeof(int));
    dv_2_y = (int*)malloc(N*sizeof(int));
    dv_2_z = (int*)malloc(N*sizeof(int));

    dv_3_x = (int*)malloc(N*sizeof(int));
    dv_3_y = (int*)malloc(N*sizeof(int));
    dv_3_z = (int*)malloc(N*sizeof(int));

    dv_1pr_x = (int*)malloc(N*sizeof(int));
    dv_1pr_y = (int*)malloc(N*sizeof(int));
    dv_1pr_z = (int*)malloc(N*sizeof(int));

    dv_2pr_x = (int*)malloc(N*sizeof(int));
    dv_2pr_y = (int*)malloc(N*sizeof(int));
    dv_2pr_z = (int*)malloc(N*sizeof(int));

    dv_3pr_x = (int*)malloc(N*sizeof(int));
    dv_3pr_y = (int*)malloc(N*sizeof(int));
    dv_3pr_z = (int*)malloc(N*sizeof(int));

    v1_x = (int*)malloc(N*sizeof(int));
    v1_y = (int*)malloc(N*sizeof(int));
    v1_z = (int*)malloc(N*sizeof(int));

    v2_x = (int*)malloc(N*sizeof(int));
    v2_y = (int*)malloc(N*sizeof(int));
    v2_z = (int*)malloc(N*sizeof(int));

    v3_x = (int*)malloc(N*sizeof(int));
    v3_y = (int*)malloc(N*sizeof(int));
    v3_z = (int*)malloc(N*sizeof(int));

    v1_prime_x = (int*)malloc(N*sizeof(int));  
    v1_prime_y = (int*)malloc(N*sizeof(int));
    v1_prime_z = (int*)malloc(N*sizeof(int));

    v2_prime_x = (int*)malloc(N*sizeof(int));  
    v2_prime_y = (int*)malloc(N*sizeof(int));
    v2_prime_z = (int*)malloc(N*sizeof(int));

    v3_prime_x = (int*)malloc(N*sizeof(int));  
    v3_prime_y = (int*)malloc(N*sizeof(int));
    v3_prime_z = (int*)malloc(N*sizeof(int));

    nv1_x = (int*)malloc(N*sizeof(int));
    nv1_y = (int*)malloc(N*sizeof(int));
    nv1_z = (int*)malloc(N*sizeof(int));

    nv2_x = (int*)malloc(N*sizeof(int));
    nv2_y = (int*)malloc(N*sizeof(int));
    nv2_z = (int*)malloc(N*sizeof(int));

    nv3_x = (int*)malloc(N*sizeof(int));
    nv3_y = (int*)malloc(N*sizeof(int));
    nv3_z = (int*)malloc(N*sizeof(int));

    nv1_prime_x = (int*)malloc(N*sizeof(int));
    nv1_prime_y = (int*)malloc(N*sizeof(int));
    nv1_prime_z = (int*)malloc(N*sizeof(int));

    nv2_prime_x = (int*)malloc(N*sizeof(int));
    nv2_prime_y = (int*)malloc(N*sizeof(int));
    nv2_prime_z = (int*)malloc(N*sizeof(int));
    
    nv3_prime_x = (int*)malloc(N*sizeof(int));
    nv3_prime_y = (int*)malloc(N*sizeof(int));
    nv3_prime_z = (int*)malloc(N*sizeof(int));

    still_together = (bool*)malloc(N*sizeof(bool));
    times = (int*)malloc(N*sizeof(int));
    not_diverged = (bool*)malloc(N*sizeof(bool));  

    cudaMalloc(&d_p1_x, N*sizeof(int)); 
    cudaMalloc(&d_p1_y, N*sizeof(int)); 
    cudaMalloc(&d_p1_z, N*sizeof(int)); 

    cudaMalloc(&d_p2_x, N*sizeof(int));
    cudaMalloc(&d_p2_y, N*sizeof(int));
    cudaMalloc(&d_p2_z, N*sizeof(int));

    cudaMalloc(&d_p3_x, N*sizeof(int));
    cudaMalloc(&d_p3_y, N*sizeof(int));
    cudaMalloc(&d_p3_z, N*sizeof(int));

    cudaMalloc(&d_p1_prime_x, N*sizeof(int));
    cudaMalloc(&d_p1_prime_y, N*sizeof(int));
    cudaMalloc(&d_p1_prime_z, N*sizeof(int));

    cudaMalloc(&d_p2_prime_x, N*sizeof(int));
    cudaMalloc(&d_p2_prime_y, N*sizeof(int));
    cudaMalloc(&d_p2_prime_z, N*sizeof(int));

    cudaMalloc(&d_p3_prime_x, N*sizeof(int));
    cudaMalloc(&d_p3_prime_y, N*sizeof(int));
    cudaMalloc(&d_p3_prime_z, N*sizeof(int));

    cudaMalloc(&d_dv_1_x, N*sizeof(int));
    cudaMalloc(&d_dv_1_y, N*sizeof(int));
    cudaMalloc(&d_dv_1_z, N*sizeof(int));

    cudaMalloc(&d_dv_2_x, N*sizeof(int));
    cudaMalloc(&d_dv_2_y, N*sizeof(int));
    cudaMalloc(&d_dv_2_z, N*sizeof(int));

    cudaMalloc(&d_dv_3_x, N*sizeof(int));
    cudaMalloc(&d_dv_3_y, N*sizeof(int));
    cudaMalloc(&d_dv_3_z, N*sizeof(int));

    cudaMalloc(&d_dv_1pr_x, N*sizeof(int));
    cudaMalloc(&d_dv_1pr_y, N*sizeof(int));
    cudaMalloc(&d_dv_1pr_z, N*sizeof(int));

    cudaMalloc(&d_dv_2pr_x, N*sizeof(int));
    cudaMalloc(&d_dv_2pr_y, N*sizeof(int));
    cudaMalloc(&d_dv_2pr_z, N*sizeof(int));

    cudaMalloc(&d_dv_3pr_x, N*sizeof(int));
    cudaMalloc(&d_dv_3pr_y, N*sizeof(int));
    cudaMalloc(&d_dv_3pr_z, N*sizeof(int));

    cudaMalloc(&d_v1_x, N*sizeof(int));
    cudaMalloc(&d_v1_y, N*sizeof(int));
    cudaMalloc(&d_v1_z, N*sizeof(int));

    cudaMalloc(&d_v2_x, N*sizeof(int));
    cudaMalloc(&d_v2_y, N*sizeof(int));
    cudaMalloc(&d_v2_z, N*sizeof(int));

    cudaMalloc(&d_v3_x, N*sizeof(int));
    cudaMalloc(&d_v3_y, N*sizeof(int));
    cudaMalloc(&d_v3_z, N*sizeof(int));

    cudaMalloc(&d_v1_prime_x, N*sizeof(int));
    cudaMalloc(&d_v1_prime_y, N*sizeof(int));
    cudaMalloc(&d_v1_prime_z, N*sizeof(int));

    cudaMalloc(&d_v2_prime_x, N*sizeof(int));
    cudaMalloc(&d_v2_prime_y, N*sizeof(int));
    cudaMalloc(&d_v2_prime_z, N*sizeof(int));

    cudaMalloc(&d_v3_prime_x, N*sizeof(int));
    cudaMalloc(&d_v3_prime_y, N*sizeof(int));
    cudaMalloc(&d_v3_prime_z, N*sizeof(int));

    cudaMalloc(&d_nv1_x, N*sizeof(int));
    cudaMalloc(&d_nv1_y, N*sizeof(int));
    cudaMalloc(&d_nv1_z, N*sizeof(int));

    cudaMalloc(&d_nv2_x, N*sizeof(int));
    cudaMalloc(&d_nv2_y, N*sizeof(int));
    cudaMalloc(&d_nv2_z, N*sizeof(int));

    cudaMalloc(&d_nv3_x, N*sizeof(int));
    cudaMalloc(&d_nv3_y, N*sizeof(int));
    cudaMalloc(&d_nv3_z, N*sizeof(int));

    cudaMalloc(&d_nv1_prime_x, N*sizeof(int));
    cudaMalloc(&d_nv1_prime_y, N*sizeof(int));
    cudaMalloc(&d_nv1_prime_z, N*sizeof(int));

    cudaMalloc(&d_nv2_prime_x, N*sizeof(int));
    cudaMalloc(&d_nv2_prime_y, N*sizeof(int));
    cudaMalloc(&d_nv2_prime_z, N*sizeof(int));

    cudaMalloc(&d_nv3_prime_x, N*sizeof(int));
    cudaMalloc(&d_nv3_prime_y, N*sizeof(int));
    cudaMalloc(&d_nv3_prime_z, N*sizeof(int));

    cudaMalloc(&d_still_together, N*sizeof(bool));
    cudaMalloc(&d_times, N*sizeof(int));
    cudaMalloc(&d_not_diverged, N*sizeof(bool));

    int resolution = sqrt(N);
    int range = 40;
    int step_size = range / resolution;
    for (int i = 0; i < N; i++) {
      int remainder = i % resolution;
      int step = i / resolution;
      p1_x[i] = -(20 << SHIFT_AMOUNT) + 40*(int(remainder)/int(resolution)) << SHIFT_AMOUNT;
      p1_y[i] = -(20 << SHIFT_AMOUNT) + 40*(int(step)/int(resolution)) << SHIFT_AMOUNT;
      p1_z[i] = -(11 << SHIFT_AMOUNT);

      p2_x[i] = 0 << SHIFT_AMOUNT;
      p2_y[i] = 0 << SHIFT_AMOUNT;
      p2_z[i] = 0 << SHIFT_AMOUNT;

      p3_x[i] = 10 << SHIFT_AMOUNT;
      p3_y[i] = 10 << SHIFT_AMOUNT;
      p3_z[i] = 12 << SHIFT_AMOUNT;

      // shift p1 in all x, y, z vectors
      p1_prime_x[i] = p1_x[i] + (1 << SHIFT_AMOUNT) / 1000;
      p1_prime_y[i] = p1_y[i] + (1 << SHIFT_AMOUNT) / 1000;
      p1_prime_z[i] = p1_z[i] + (1 << SHIFT_AMOUNT) / 1000;

      p2_prime_x[i] = 0;
      p2_prime_y[i] = 0;
      p2_prime_z[i] = 0;

      p3_prime_x[i] = 10 << SHIFT_AMOUNT;
      p3_prime_y[i] = 10 << SHIFT_AMOUNT;
      p3_prime_z[i] = 12 << SHIFT_AMOUNT;

      v1_x[i] = -(3 << SHIFT_AMOUNT);
      v1_y[i] = 0;
      v1_z[i] = 0;

      v2_x[i] = 0;
      v2_y[i] = 0;
      v2_z[i] = 0;

      v3_x[i] = 3 << SHIFT_AMOUNT;
      v3_y[i] = 0;
      v3_z[i] = 0;

      v1_prime_x[i] = -(3 << SHIFT_AMOUNT);
      v1_prime_y[i] = 0;
      v1_prime_z[i] = 0;

      v2_prime_x[i] = 0;
      v2_prime_y[i] = 0;
      v2_prime_z[i] = 0;

      v3_prime_x[i] = 3 << SHIFT_AMOUNT;
      v3_prime_y[i] = 0;
      v3_prime_z[i] = 0;

      times[i] = 0;
      still_together[i] = true;
      not_diverged[i] = true;
    }

    cudaMemcpy(d_p1_x, p1_x, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p1_y, p1_y, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p1_z, p1_z, N*sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_p2_x, p2_x, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p2_y, p2_y, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p2_z, p2_z, N*sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_p3_x, p3_x, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p3_y, p3_y, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p3_z, p3_z, N*sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_p1_prime_x, p1_prime_x, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p1_prime_y, p1_prime_y, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p1_prime_z, p1_prime_z, N*sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_p2_prime_x, p2_prime_x, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p2_prime_y, p2_prime_y, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p2_prime_z, p2_prime_z, N*sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_p3_prime_x, p3_prime_x, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p3_prime_y, p3_prime_y, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p3_prime_z, p3_prime_z, N*sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_dv_1_x, dv_1_x, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dv_1_y, dv_1_y, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dv_1_z, dv_1_z, N*sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_dv_2_x, dv_2_x, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dv_2_y, dv_2_y, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dv_2_z, dv_2_z, N*sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_dv_3_x, dv_3_x, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dv_3_y, dv_3_y, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dv_3_z, dv_3_z, N*sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_dv_1pr_x, dv_1pr_x, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dv_1pr_y, dv_1pr_y, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dv_1pr_z, dv_1pr_z, N*sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_dv_2pr_x, dv_2pr_x, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dv_2pr_y, dv_2pr_y, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dv_2pr_z, dv_2pr_z, N*sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_dv_3pr_x, dv_3pr_x, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dv_3pr_y, dv_3pr_y, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dv_3pr_z, dv_3pr_z, N*sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_v1_x, v1_x, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v1_y, v1_y, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v1_z, v1_z, N*sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_v2_x, v2_x, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v2_y, v2_y, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v2_z, v2_z, N*sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_v3_x, v3_x, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v3_y, v3_y, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v3_z, v3_z, N*sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_v1_prime_x, v1_prime_x, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v1_prime_y, v1_prime_y, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v1_prime_z, v1_prime_z, N*sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_v2_prime_x, v2_prime_x, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v2_prime_y, v2_prime_y, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v2_prime_z, v2_prime_z, N*sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_v3_prime_x, v3_prime_x, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v3_prime_y, v3_prime_y, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v3_prime_z, v3_prime_z, N*sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_nv1_x, nv1_x, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nv1_y, nv1_y, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nv1_z, nv1_z, N*sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_nv2_x, nv2_x, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nv2_y, nv2_y, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nv2_z, nv2_z, N*sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_nv3_x, nv3_x, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nv3_y, nv3_y, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nv3_z, nv3_z, N*sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_nv1_prime_x, nv1_prime_x, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nv1_prime_y, nv1_prime_y, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nv1_prime_z, nv1_prime_z, N*sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_nv2_prime_x, nv2_prime_x, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nv2_prime_y, nv2_prime_y, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nv2_prime_z, nv2_prime_z, N*sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_nv3_prime_x, nv3_prime_x, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nv3_prime_y, nv3_prime_y, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nv3_prime_z, nv3_prime_z, N*sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_times, times, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_still_together, still_together, N*sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(d_not_diverged, not_diverged, N*sizeof(bool), cudaMemcpyHostToDevice);

    // initialize timer
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();

    // call CUDA kernal on inputs in configuration <<< blockIdx, threadIdx>>>>
    divergence<<<(N+63)/64, 64>>>(
        N, 
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

    cudaDeviceSynchronize();

    // measure elapsed kernal runtime
    end = std::chrono::system_clock::now();
    std::chrono::duration<float> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Elapsed Time: " << elapsed_seconds.count() << "s\n";

    cudaMemcpy(times, d_times, N*sizeof(int), cudaMemcpyDeviceToHost);
    for (int i=0; i<20; i++){
      std::cout << times[i] << '\n';
    }

    cudaFree(d_p1_x); cudaFree(d_p1_y); cudaFree(d_p1_z);
    cudaFree(d_p2_x); cudaFree(d_p2_y); cudaFree(d_p2_z);
    cudaFree(d_p3_x); cudaFree(d_p3_y); cudaFree(d_p3_z);

    cudaFree(d_p1_prime_x); cudaFree(d_p1_prime_y); cudaFree(d_p1_prime_z);  
    cudaFree(d_p2_prime_x); cudaFree(d_p2_prime_y); cudaFree(d_p2_prime_z);
    cudaFree(d_p3_prime_x); cudaFree(d_p3_prime_y); cudaFree(d_p3_prime_z);

    cudaFree(d_dv_1_x); cudaFree(d_dv_1_y); cudaFree(d_dv_1_z);
    cudaFree(d_dv_2_x); cudaFree(d_dv_2_y); cudaFree(d_dv_2_z);
    cudaFree(d_dv_3_x); cudaFree(d_dv_3_y); cudaFree(d_dv_3_z);

    cudaFree(d_dv_1pr_x); cudaFree(d_dv_1pr_y); cudaFree(d_dv_1pr_z);
    cudaFree(d_dv_2pr_x); cudaFree(d_dv_2pr_y); cudaFree(d_dv_2pr_z);
    cudaFree(d_dv_3pr_x); cudaFree(d_dv_3pr_y); cudaFree(d_dv_3pr_z);

    cudaFree(d_nv1_x); cudaFree(d_nv1_y); cudaFree(d_nv1_z);
    cudaFree(d_nv2_x); cudaFree(d_nv2_y); cudaFree(d_nv2_z);
    cudaFree(d_nv3_x); cudaFree(d_nv3_y); cudaFree(d_nv3_z);

    cudaFree(d_nv1_prime_x); cudaFree(d_nv1_prime_y); cudaFree(d_nv1_prime_z);
    cudaFree(d_nv2_prime_x); cudaFree(d_nv2_prime_y); cudaFree(d_nv2_prime_z);
    cudaFree(d_nv3_prime_x); cudaFree(d_nv3_prime_y); cudaFree(d_nv3_prime_z);

    cudaFree(d_times); cudaFree(d_still_together); cudaFree(d_not_diverged);

    free(p1_x); free(p1_y); free(p1_z);
    free(p2_x); free(p2_y); free(p2_z);
    free(p3_x); free(p3_y); free(p3_z);

    free(p1_prime_x); free(p1_prime_y); free(p1_prime_z);  
    free(p2_prime_x); free(p2_prime_y); free(p2_prime_z);  
    free(p3_prime_x); free(p3_prime_y); free(p3_prime_z);  

    free(dv_1_x); free(dv_1_y); free(dv_1_z);
    free(dv_2_x); free(dv_2_y); free(dv_2_z);
    free(dv_3_x); free(dv_3_y); free(dv_3_z);

    free(dv_1pr_x); free(dv_1pr_y); free(dv_1pr_z);
    free(dv_2pr_x); free(dv_2pr_y); free(dv_2pr_z);
    free(dv_3pr_x); free(dv_3pr_y); free(dv_3pr_z);

    free(nv1_x); free(nv1_y); free(nv1_z);
    free(nv2_x); free(nv2_y); free(nv2_z);
    free(nv3_x); free(nv3_y); free(nv3_z);

    free(nv1_prime_x); free(nv1_prime_y); free(nv1_prime_z);
    free(nv2_prime_x); free(nv2_prime_y); free(nv2_prime_z);
    free(nv3_prime_x); free(nv3_prime_y); free(nv3_prime_z);

    free(still_together); free(not_diverged);
    return 0;

  
}














