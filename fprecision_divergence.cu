#include <stdio.h>
#include <iostream>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime_api.h>



// kernal declaration
__global__
void divergence(uint n, 
              uint steps,
              uint delta_t,
              bool *still_together,
              bool *not_diverged,
              uint *times,
              uint m_1, uint m_2, uint m_3,
              uint critical_distance,
              uint *p1_x, uint *p1_y, uint *p1_z, 
              uint *p2_x, uint *p2_y, uint *p2_z, 
              uint *p3_x, uint *p3_y, uint *p3_z, 
              uint *p1_prime_x, uint *p1_prime_y, uint *p1_prime_z, 
              uint *p2_prime_x, uint *p2_prime_y, uint *p2_prime_z, 
              uint *p3_prime_x, uint *p3_prime_y, uint *p3_prime_z, 
              uint *dv_1_x, uint *dv_1_y, uint *dv_1_z,
              uint *dv_2_x, uint *dv_2_y, uint *dv_2_z,
              uint *dv_3_x, uint *dv_3_y, uint *dv_3_z,
              uint *dv_1pr_x, uint *dv_1pr_y, uint *dv_1pr_z,
              uint *dv_2pr_x, uint *dv_2pr_y, uint *dv_2pr_z,
              uint *dv_3pr_x, uint *dv_3pr_y, uint *dv_3pr_z,
              uint *v1_x, uint *v1_y, uint *v1_z,
              uint *v2_x, uint *v2_y, uint *v2_z,
              uint *v3_x, uint *v3_y, uint *v3_z,
              uint *v1_prime_x, uint *v1_prime_y, uint *v1_prime_z,
              uint *v2_prime_x, uint *v2_prime_y, uint *v2_prime_z,
              uint *v3_prime_x, uint *v3_prime_y, uint *v3_prime_z,
              uint *nv1_x, uint *nv1_y, uint *nv1_z,
              uint *nv2_x, uint *nv2_y, uint *nv2_z,
              uint *nv3_x, uint *nv3_y, uint *nv3_z,
              uint *nv1_prime_x, uint *nv1_prime_y, uint *nv1_prime_z,
              uint *nv2_prime_x, uint *nv2_prime_y, uint *nv2_prime_z,
              uint *nv3_prime_x, uint *nv3_prime_y, uint *nv3_prime_z
              )
{
  uint i = blockIdx.x*blockDim.x + threadIdx.x;
  for (uint j=0; j < steps; j++) {
    if (i < n){
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

uint main(void){
    uint N = 90000;
    uint steps = 50000;
    float delta_t = 0.001;
    float critical_distance = 0.5;
    uint m1 = 10.;
    uint m2 = 20.;
    uint m3 = 30.;
    uint *p1_x, *p1_y, *p1_z;
    uint *p2_x, *p2_y, *p2_z;
    uint *p3_x, *p3_y, *p3_z;
    uint *p1_prime_x, *p1_prime_y, *p1_prime_z;
    uint *p2_prime_x, *p2_prime_y, *p2_prime_z;
    uint *p3_prime_x, *p3_prime_y, *p3_prime_z;
    uint *dv_1_x, *dv_1_y, *dv_1_z;
    uint *dv_2_x, *dv_2_y, *dv_2_z;
    uint *dv_3_x, *dv_3_y, *dv_3_z;
    uint *dv_1pr_x, *dv_1pr_y, *dv_1pr_z;
    uint *dv_2pr_x, *dv_2pr_y, *dv_2pr_z;
    uint *dv_3pr_x, *dv_3pr_y, *dv_3pr_z;
    uint *v1_x, *v1_y, *v1_z;
    uint *v2_x, *v2_y, *v2_z;
    uint *v3_x, *v3_y, *v3_z;
    uint *v1_prime_x, *v1_prime_y, *v1_prime_z;
    uint *v2_prime_x, *v2_prime_y, *v2_prime_z;
    uint *v3_prime_x, *v3_prime_y, *v3_prime_z;
    uint *nv1_x, *nv1_y, *nv1_z;
    uint *nv2_x, *nv2_y, *nv2_z;
    uint *nv3_x, *nv3_y, *nv3_z;
    uint *nv1_prime_x, *nv1_prime_y, *nv1_prime_z;
    uint *nv2_prime_x, *nv2_prime_y, *nv2_prime_z;
    uint *nv3_prime_x, *nv3_prime_y, *nv3_prime_z;

    uint *d_p1_x, *d_p1_y, *d_p1_z;
    uint *d_p2_x, *d_p2_y, *d_p2_z;
    uint *d_p3_x, *d_p3_y, *d_p3_z;
    uint *d_p1_prime_x, *d_p1_prime_y, *d_p1_prime_z;
    uint *d_p2_prime_x, *d_p2_prime_y, *d_p2_prime_z;
    uint *d_p3_prime_x, *d_p3_prime_y, *d_p3_prime_z;
    uint *d_dv_1_x, *d_dv_1_y, *d_dv_1_z;
    uint *d_dv_2_x, *d_dv_2_y, *d_dv_2_z;
    uint *d_dv_3_x, *d_dv_3_y, *d_dv_3_z;
    uint *d_dv_1pr_x, *d_dv_1pr_y, *d_dv_1pr_z;
    uint *d_dv_2pr_x, *d_dv_2pr_y, *d_dv_2pr_z;
    uint *d_dv_3pr_x, *d_dv_3pr_y, *d_dv_3pr_z;
    uint *d_v1_x, *d_v1_y, *d_v1_z;
    uint *d_v2_x, *d_v2_y, *d_v2_z;
    uint *d_v3_x, *d_v3_y, *d_v3_z;
    uint *d_v1_prime_x, *d_v1_prime_y, *d_v1_prime_z;
    uint *d_v2_prime_x, *d_v2_prime_y, *d_v2_prime_z;
    uint *d_v3_prime_x, *d_v3_prime_y, *d_v3_prime_z;
    uint *d_nv1_x, *d_nv1_y, *d_nv1_z;
    uint *d_nv2_x, *d_nv2_y, *d_nv2_z;
    uint *d_nv3_x, *d_nv3_y, *d_nv3_z;
    uint *d_nv1_prime_x, *d_nv1_prime_y, *d_nv1_prime_z;
    uint *d_nv2_prime_x, *d_nv2_prime_y, *d_nv2_prime_z;
    uint *d_nv3_prime_x, *d_nv3_prime_y, *d_nv3_prime_z;

    bool *still_together, *d_still_together;
    uint *times, *d_times;
    bool *not_diverged, *d_not_diverged;

    p1_x = (uint*)malloc(N*sizeof(uint));
    p1_y = (uint*)malloc(N*sizeof(uint));
    p1_z = (uint*)malloc(N*sizeof(uint));

    p2_x = (uint*)malloc(N*sizeof(uint));
    p2_y = (uint*)malloc(N*sizeof(uint));
    p2_z = (uint*)malloc(N*sizeof(uint));

    p3_x = (uint*)malloc(N*sizeof(uint));
    p3_y = (uint*)malloc(N*sizeof(uint));
    p3_z = (uint*)malloc(N*sizeof(uint));

    p1_prime_x = (uint*)malloc(N*sizeof(uint));
    p1_prime_y = (uint*)malloc(N*sizeof(uint));
    p1_prime_z = (uint*)malloc(N*sizeof(uint));

    p2_prime_x = (uint*)malloc(N*sizeof(uint));
    p2_prime_y = (uint*)malloc(N*sizeof(uint));
    p2_prime_z = (uint*)malloc(N*sizeof(uint));

    p3_prime_x = (uint*)malloc(N*sizeof(uint));
    p3_prime_y = (uint*)malloc(N*sizeof(uint));
    p3_prime_z = (uint*)malloc(N*sizeof(uint));

    dv_1_x = (uint*)malloc(N*sizeof(uint));
    dv_1_y = (uint*)malloc(N*sizeof(uint));
    dv_1_z = (uint*)malloc(N*sizeof(uint));

    dv_2_x = (uint*)malloc(N*sizeof(uint));
    dv_2_y = (uint*)malloc(N*sizeof(uint));
    dv_2_z = (uint*)malloc(N*sizeof(uint));

    dv_3_x = (uint*)malloc(N*sizeof(uint));
    dv_3_y = (uint*)malloc(N*sizeof(uint));
    dv_3_z = (uint*)malloc(N*sizeof(uint));

    dv_1pr_x = (uint*)malloc(N*sizeof(uint));
    dv_1pr_y = (uint*)malloc(N*sizeof(uint));
    dv_1pr_z = (uint*)malloc(N*sizeof(uint));

    dv_2pr_x = (uint*)malloc(N*sizeof(uint));
    dv_2pr_y = (uint*)malloc(N*sizeof(uint));
    dv_2pr_z = (uint*)malloc(N*sizeof(uint));

    dv_3pr_x = (uint*)malloc(N*sizeof(uint));
    dv_3pr_y = (uint*)malloc(N*sizeof(uint));
    dv_3pr_z = (uint*)malloc(N*sizeof(uint));

    v1_x = (uint*)malloc(N*sizeof(uint));
    v1_y = (uint*)malloc(N*sizeof(uint));
    v1_z = (uint*)malloc(N*sizeof(uint));

    v2_x = (uint*)malloc(N*sizeof(uint));
    v2_y = (uint*)malloc(N*sizeof(uint));
    v2_z = (uint*)malloc(N*sizeof(uint));

    v3_x = (uint*)malloc(N*sizeof(uint));
    v3_y = (uint*)malloc(N*sizeof(uint));
    v3_z = (uint*)malloc(N*sizeof(uint));

    v1_prime_x = (uint*)malloc(N*sizeof(uint));  
    v1_prime_y = (uint*)malloc(N*sizeof(uint));
    v1_prime_z = (uint*)malloc(N*sizeof(uint));

    v2_prime_x = (uint*)malloc(N*sizeof(uint));  
    v2_prime_y = (uint*)malloc(N*sizeof(uint));
    v2_prime_z = (uint*)malloc(N*sizeof(uint));

    v3_prime_x = (uint*)malloc(N*sizeof(uint));  
    v3_prime_y = (uint*)malloc(N*sizeof(uint));
    v3_prime_z = (uint*)malloc(N*sizeof(uint));

    nv1_x = (uint*)malloc(N*sizeof(uint));
    nv1_y = (uint*)malloc(N*sizeof(uint));
    nv1_z = (uint*)malloc(N*sizeof(uint));

    nv2_x = (uint*)malloc(N*sizeof(uint));
    nv2_y = (uint*)malloc(N*sizeof(uint));
    nv2_z = (uint*)malloc(N*sizeof(uint));

    nv3_x = (uint*)malloc(N*sizeof(uint));
    nv3_y = (uint*)malloc(N*sizeof(uint));
    nv3_z = (uint*)malloc(N*sizeof(uint));

    nv1_prime_x = (uint*)malloc(N*sizeof(uint));
    nv1_prime_y = (uint*)malloc(N*sizeof(uint));
    nv1_prime_z = (uint*)malloc(N*sizeof(uint));

    nv2_prime_x = (uint*)malloc(N*sizeof(uint));
    nv2_prime_y = (uint*)malloc(N*sizeof(uint));
    nv2_prime_z = (uint*)malloc(N*sizeof(uint));
    
    nv3_prime_x = (uint*)malloc(N*sizeof(uint));
    nv3_prime_y = (uint*)malloc(N*sizeof(uint));
    nv3_prime_z = (uint*)malloc(N*sizeof(uint));

    still_together = (bool*)malloc(N*sizeof(bool));
    times = (uint*)malloc(N*sizeof(uint));
    not_diverged = (bool*)malloc(N*sizeof(bool));  

    cudaMalloc(&d_p1_x, N*sizeof(uint)); 
    cudaMalloc(&d_p1_y, N*sizeof(uint)); 
    cudaMalloc(&d_p1_z, N*sizeof(uint)); 

    cudaMalloc(&d_p2_x, N*sizeof(uint));
    cudaMalloc(&d_p2_y, N*sizeof(uint));
    cudaMalloc(&d_p2_z, N*sizeof(uint));

    cudaMalloc(&d_p3_x, N*sizeof(uint));
    cudaMalloc(&d_p3_y, N*sizeof(uint));
    cudaMalloc(&d_p3_z, N*sizeof(uint));

    cudaMalloc(&d_p1_prime_x, N*sizeof(uint));
    cudaMalloc(&d_p1_prime_y, N*sizeof(uint));
    cudaMalloc(&d_p1_prime_z, N*sizeof(uint));

    cudaMalloc(&d_p2_prime_x, N*sizeof(uint));
    cudaMalloc(&d_p2_prime_y, N*sizeof(uint));
    cudaMalloc(&d_p2_prime_z, N*sizeof(uint));

    cudaMalloc(&d_p3_prime_x, N*sizeof(uint));
    cudaMalloc(&d_p3_prime_y, N*sizeof(uint));
    cudaMalloc(&d_p3_prime_z, N*sizeof(uint));

    cudaMalloc(&d_dv_1_x, N*sizeof(uint));
    cudaMalloc(&d_dv_1_y, N*sizeof(uint));
    cudaMalloc(&d_dv_1_z, N*sizeof(uint));

    cudaMalloc(&d_dv_2_x, N*sizeof(uint));
    cudaMalloc(&d_dv_2_y, N*sizeof(uint));
    cudaMalloc(&d_dv_2_z, N*sizeof(uint));

    cudaMalloc(&d_dv_3_x, N*sizeof(uint));
    cudaMalloc(&d_dv_3_y, N*sizeof(uint));
    cudaMalloc(&d_dv_3_z, N*sizeof(uint));

    cudaMalloc(&d_dv_1pr_x, N*sizeof(uint));
    cudaMalloc(&d_dv_1pr_y, N*sizeof(uint));
    cudaMalloc(&d_dv_1pr_z, N*sizeof(uint));

    cudaMalloc(&d_dv_2pr_x, N*sizeof(uint));
    cudaMalloc(&d_dv_2pr_y, N*sizeof(uint));
    cudaMalloc(&d_dv_2pr_z, N*sizeof(uint));

    cudaMalloc(&d_dv_3pr_x, N*sizeof(uint));
    cudaMalloc(&d_dv_3pr_y, N*sizeof(uint));
    cudaMalloc(&d_dv_3pr_z, N*sizeof(uint));

    cudaMalloc(&d_v1_x, N*sizeof(uint));
    cudaMalloc(&d_v1_y, N*sizeof(uint));
    cudaMalloc(&d_v1_z, N*sizeof(uint));

    cudaMalloc(&d_v2_x, N*sizeof(uint));
    cudaMalloc(&d_v2_y, N*sizeof(uint));
    cudaMalloc(&d_v2_z, N*sizeof(uint));

    cudaMalloc(&d_v3_x, N*sizeof(uint));
    cudaMalloc(&d_v3_y, N*sizeof(uint));
    cudaMalloc(&d_v3_z, N*sizeof(uint));

    cudaMalloc(&d_v1_prime_x, N*sizeof(uint));
    cudaMalloc(&d_v1_prime_y, N*sizeof(uint));
    cudaMalloc(&d_v1_prime_z, N*sizeof(uint));

    cudaMalloc(&d_v2_prime_x, N*sizeof(uint));
    cudaMalloc(&d_v2_prime_y, N*sizeof(uint));
    cudaMalloc(&d_v2_prime_z, N*sizeof(uint));

    cudaMalloc(&d_v3_prime_x, N*sizeof(uint));
    cudaMalloc(&d_v3_prime_y, N*sizeof(uint));
    cudaMalloc(&d_v3_prime_z, N*sizeof(uint));

    cudaMalloc(&d_nv1_x, N*sizeof(uint));
    cudaMalloc(&d_nv1_y, N*sizeof(uint));
    cudaMalloc(&d_nv1_z, N*sizeof(uint));

    cudaMalloc(&d_nv2_x, N*sizeof(uint));
    cudaMalloc(&d_nv2_y, N*sizeof(uint));
    cudaMalloc(&d_nv2_z, N*sizeof(uint));

    cudaMalloc(&d_nv3_x, N*sizeof(uint));
    cudaMalloc(&d_nv3_y, N*sizeof(uint));
    cudaMalloc(&d_nv3_z, N*sizeof(uint));

    cudaMalloc(&d_nv1_prime_x, N*sizeof(uint));
    cudaMalloc(&d_nv1_prime_y, N*sizeof(uint));
    cudaMalloc(&d_nv1_prime_z, N*sizeof(uint));

    cudaMalloc(&d_nv2_prime_x, N*sizeof(uint));
    cudaMalloc(&d_nv2_prime_y, N*sizeof(uint));
    cudaMalloc(&d_nv2_prime_z, N*sizeof(uint));

    cudaMalloc(&d_nv3_prime_x, N*sizeof(uint));
    cudaMalloc(&d_nv3_prime_y, N*sizeof(uint));
    cudaMalloc(&d_nv3_prime_z, N*sizeof(uint));

    cudaMalloc(&d_still_together, N*sizeof(bool));
    cudaMalloc(&d_times, N*sizeof(uint));
    cudaMalloc(&d_not_diverged, N*sizeof(bool));

    uint resolution = sqrt(N);
    uint range = 40;
    uint step_size = range / resolution;
    for (uint i = 0; i < N; i++) {
      uint remainder = i % resolution;
      uint step = i / resolution;
      p1_x[i] = -20. + 40*(uint(remainder)/uint(resolution));
      p1_y[i] = -20. + 40*(uint(step)/uint(resolution));
      p1_z[i] = -11.0;

      p2_x[i] = 0.0;
      p2_y[i] = 0.0;
      p2_z[i] = 0.0;

      p3_x[i] = 10.0;
      p3_y[i] = 10.0;
      p3_z[i] = 12.0;

      // shift p1 in all x, y, z vectors
      p1_prime_x[i] = -20. + 40*(uint(remainder)/uint(resolution)) + 0.001;
      p1_prime_y[i] = -20. + 40*(uint(step)/uint(resolution)) + 0.001;
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

    cudaMemcpy(d_p1_x, p1_x, N*sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p1_y, p1_y, N*sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p1_z, p1_z, N*sizeof(uint), cudaMemcpyHostToDevice);

    cudaMemcpy(d_p2_x, p2_x, N*sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p2_y, p2_y, N*sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p2_z, p2_z, N*sizeof(uint), cudaMemcpyHostToDevice);

    cudaMemcpy(d_p3_x, p3_x, N*sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p3_y, p3_y, N*sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p3_z, p3_z, N*sizeof(uint), cudaMemcpyHostToDevice);

    cudaMemcpy(d_p1_prime_x, p1_prime_x, N*sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p1_prime_y, p1_prime_y, N*sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p1_prime_z, p1_prime_z, N*sizeof(uint), cudaMemcpyHostToDevice);

    cudaMemcpy(d_p2_prime_x, p2_prime_x, N*sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p2_prime_y, p2_prime_y, N*sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p2_prime_z, p2_prime_z, N*sizeof(uint), cudaMemcpyHostToDevice);

    cudaMemcpy(d_p3_prime_x, p3_prime_x, N*sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p3_prime_y, p3_prime_y, N*sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p3_prime_z, p3_prime_z, N*sizeof(uint), cudaMemcpyHostToDevice);

    cudaMemcpy(d_dv_1_x, dv_1_x, N*sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dv_1_y, dv_1_y, N*sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dv_1_z, dv_1_z, N*sizeof(uint), cudaMemcpyHostToDevice);

    cudaMemcpy(d_dv_2_x, dv_2_x, N*sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dv_2_y, dv_2_y, N*sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dv_2_z, dv_2_z, N*sizeof(uint), cudaMemcpyHostToDevice);

    cudaMemcpy(d_dv_3_x, dv_3_x, N*sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dv_3_y, dv_3_y, N*sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dv_3_z, dv_3_z, N*sizeof(uint), cudaMemcpyHostToDevice);

    cudaMemcpy(d_dv_1pr_x, dv_1pr_x, N*sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dv_1pr_y, dv_1pr_y, N*sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dv_1pr_z, dv_1pr_z, N*sizeof(uint), cudaMemcpyHostToDevice);

    cudaMemcpy(d_dv_2pr_x, dv_2pr_x, N*sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dv_2pr_y, dv_2pr_y, N*sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dv_2pr_z, dv_2pr_z, N*sizeof(uint), cudaMemcpyHostToDevice);

    cudaMemcpy(d_dv_3pr_x, dv_3pr_x, N*sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dv_3pr_y, dv_3pr_y, N*sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dv_3pr_z, dv_3pr_z, N*sizeof(uint), cudaMemcpyHostToDevice);

    cudaMemcpy(d_v1_x, v1_x, N*sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v1_y, v1_y, N*sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v1_z, v1_z, N*sizeof(uint), cudaMemcpyHostToDevice);

    cudaMemcpy(d_v2_x, v2_x, N*sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v2_y, v2_y, N*sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v2_z, v2_z, N*sizeof(uint), cudaMemcpyHostToDevice);

    cudaMemcpy(d_v3_x, v3_x, N*sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v3_y, v3_y, N*sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v3_z, v3_z, N*sizeof(uint), cudaMemcpyHostToDevice);

    cudaMemcpy(d_v1_prime_x, v1_prime_x, N*sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v1_prime_y, v1_prime_y, N*sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v1_prime_z, v1_prime_z, N*sizeof(uint), cudaMemcpyHostToDevice);

    cudaMemcpy(d_v2_prime_x, v2_prime_x, N*sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v2_prime_y, v2_prime_y, N*sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v2_prime_z, v2_prime_z, N*sizeof(uint), cudaMemcpyHostToDevice);

    cudaMemcpy(d_v3_prime_x, v3_prime_x, N*sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v3_prime_y, v3_prime_y, N*sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v3_prime_z, v3_prime_z, N*sizeof(uint), cudaMemcpyHostToDevice);

    cudaMemcpy(d_nv1_x, nv1_x, N*sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nv1_y, nv1_y, N*sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nv1_z, nv1_z, N*sizeof(uint), cudaMemcpyHostToDevice);

    cudaMemcpy(d_nv2_x, nv2_x, N*sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nv2_y, nv2_y, N*sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nv2_z, nv2_z, N*sizeof(uint), cudaMemcpyHostToDevice);

    cudaMemcpy(d_nv3_x, nv3_x, N*sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nv3_y, nv3_y, N*sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nv3_z, nv3_z, N*sizeof(uint), cudaMemcpyHostToDevice);

    cudaMemcpy(d_nv1_prime_x, nv1_prime_x, N*sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nv1_prime_y, nv1_prime_y, N*sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nv1_prime_z, nv1_prime_z, N*sizeof(uint), cudaMemcpyHostToDevice);

    cudaMemcpy(d_nv2_prime_x, nv2_prime_x, N*sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nv2_prime_y, nv2_prime_y, N*sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nv2_prime_z, nv2_prime_z, N*sizeof(uint), cudaMemcpyHostToDevice);

    cudaMemcpy(d_nv3_prime_x, nv3_prime_x, N*sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nv3_prime_y, nv3_prime_y, N*sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nv3_prime_z, nv3_prime_z, N*sizeof(uint), cudaMemcpyHostToDevice);

    cudaMemcpy(d_times, times, N*sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_still_together, still_together, N*sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(d_not_diverged, not_diverged, N*sizeof(bool), cudaMemcpyHostToDevice);

    // initialize timer
    std::chrono::time_pouint<std::chrono::system_clock> start, end;
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

    cudaMemcpy(times, d_times, N*sizeof(uint), cudaMemcpyDeviceToHost);

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

  
}














