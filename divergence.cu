#include <stdio.h>
#include <iostream>
#include <chrono>

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

  p1_x = (double*)malloc(N*sizeof(double));
  p1_y = (double*)malloc(N*sizeof(double));
  p1_z = (double*)malloc(N*sizeof(double));

  p2_x = (double*)malloc(N*sizeof(double));
  p2_y = (double*)malloc(N*sizeof(double));
  p2_z = (double*)malloc(N*sizeof(double));

  p3_x = (double*)malloc(N*sizeof(double));
  p3_y = (double*)malloc(N*sizeof(double));
  p3_z = (double*)malloc(N*sizeof(double));

  p1_prime_x = (double*)malloc(N*sizeof(double));
  p1_prime_y = (double*)malloc(N*sizeof(double));
  p1_prime_z = (double*)malloc(N*sizeof(double));

  p2_prime_x = (double*)malloc(N*sizeof(double));
  p2_prime_y = (double*)malloc(N*sizeof(double));
  p2_prime_z = (double*)malloc(N*sizeof(double));

  p3_prime_x = (double*)malloc(N*sizeof(double));
  p3_prime_y = (double*)malloc(N*sizeof(double));
  p3_prime_z = (double*)malloc(N*sizeof(double));

  dv_1_x = (double*)malloc(N*sizeof(double));
  dv_1_y = (double*)malloc(N*sizeof(double));
  dv_1_z = (double*)malloc(N*sizeof(double));

  dv_2_x = (double*)malloc(N*sizeof(double));
  dv_2_y = (double*)malloc(N*sizeof(double));
  dv_2_z = (double*)malloc(N*sizeof(double));

  dv_3_x = (double*)malloc(N*sizeof(double));
  dv_3_y = (double*)malloc(N*sizeof(double));
  dv_3_z = (double*)malloc(N*sizeof(double));

  dv_1pr_x = (double*)malloc(N*sizeof(double));
  dv_1pr_y = (double*)malloc(N*sizeof(double));
  dv_1pr_z = (double*)malloc(N*sizeof(double));

  dv_2pr_x = (double*)malloc(N*sizeof(double));
  dv_2pr_y = (double*)malloc(N*sizeof(double));
  dv_2pr_z = (double*)malloc(N*sizeof(double));

  dv_3pr_x = (double*)malloc(N*sizeof(double));
  dv_3pr_y = (double*)malloc(N*sizeof(double));
  dv_3pr_z = (double*)malloc(N*sizeof(double));

  v1_x = (double*)malloc(N*sizeof(double));
  v1_y = (double*)malloc(N*sizeof(double));
  v1_z = (double*)malloc(N*sizeof(double));

  v2_x = (double*)malloc(N*sizeof(double));
  v2_y = (double*)malloc(N*sizeof(double));
  v2_z = (double*)malloc(N*sizeof(double));

  v3_x = (double*)malloc(N*sizeof(double));
  v3_y = (double*)malloc(N*sizeof(double));
  v3_z = (double*)malloc(N*sizeof(double));

  v1_prime_x = (double*)malloc(N*sizeof(double));  
  v1_prime_y = (double*)malloc(N*sizeof(double));
  v1_prime_z = (double*)malloc(N*sizeof(double));

  v2_prime_x = (double*)malloc(N*sizeof(double));  
  v2_prime_y = (double*)malloc(N*sizeof(double));
  v2_prime_z = (double*)malloc(N*sizeof(double));

  v3_prime_x = (double*)malloc(N*sizeof(double));  
  v3_prime_y = (double*)malloc(N*sizeof(double));
  v3_prime_z = (double*)malloc(N*sizeof(double));

  nv1_x = (double*)malloc(N*sizeof(double));
  nv1_y = (double*)malloc(N*sizeof(double));
  nv1_z = (double*)malloc(N*sizeof(double));

  nv2_x = (double*)malloc(N*sizeof(double));
  nv2_y = (double*)malloc(N*sizeof(double));
  nv2_z = (double*)malloc(N*sizeof(double));

  nv3_x = (double*)malloc(N*sizeof(double));
  nv3_y = (double*)malloc(N*sizeof(double));
  nv3_z = (double*)malloc(N*sizeof(double));

  nv1_prime_x = (double*)malloc(N*sizeof(double));
  nv1_prime_y = (double*)malloc(N*sizeof(double));
  nv1_prime_z = (double*)malloc(N*sizeof(double));

  nv2_prime_x = (double*)malloc(N*sizeof(double));
  nv2_prime_y = (double*)malloc(N*sizeof(double));
  nv2_prime_z = (double*)malloc(N*sizeof(double));
  
  nv3_prime_x = (double*)malloc(N*sizeof(double));
  nv3_prime_y = (double*)malloc(N*sizeof(double));
  nv3_prime_z = (double*)malloc(N*sizeof(double));

  still_together = (bool*)malloc(N*sizeof(bool));
  times = (int*)malloc(N*sizeof(int));
  not_diverged = (bool*)malloc(N*sizeof(bool));  

  cudaMalloc(&d_p1_x, N*sizeof(double)); 
  cudaMalloc(&d_p1_y, N*sizeof(double)); 
  cudaMalloc(&d_p1_z, N*sizeof(double)); 

  cudaMalloc(&d_p2_x, N*sizeof(double));
  cudaMalloc(&d_p2_y, N*sizeof(double));
  cudaMalloc(&d_p2_z, N*sizeof(double));

  cudaMalloc(&d_p3_x, N*sizeof(double));
  cudaMalloc(&d_p3_y, N*sizeof(double));
  cudaMalloc(&d_p3_z, N*sizeof(double));

  cudaMalloc(&d_p1_prime_x, N*sizeof(double));
  cudaMalloc(&d_p1_prime_y, N*sizeof(double));
  cudaMalloc(&d_p1_prime_z, N*sizeof(double));

  cudaMalloc(&d_p2_prime_x, N*sizeof(double));
  cudaMalloc(&d_p2_prime_y, N*sizeof(double));
  cudaMalloc(&d_p2_prime_z, N*sizeof(double));

  cudaMalloc(&d_p3_prime_x, N*sizeof(double));
  cudaMalloc(&d_p3_prime_y, N*sizeof(double));
  cudaMalloc(&d_p3_prime_z, N*sizeof(double));

  cudaMalloc(&d_dv_1_x, N*sizeof(double));
  cudaMalloc(&d_dv_1_y, N*sizeof(double));
  cudaMalloc(&d_dv_1_z, N*sizeof(double));

  cudaMalloc(&d_dv_2_x, N*sizeof(double));
  cudaMalloc(&d_dv_2_y, N*sizeof(double));
  cudaMalloc(&d_dv_2_z, N*sizeof(double));

  cudaMalloc(&d_dv_3_x, N*sizeof(double));
  cudaMalloc(&d_dv_3_y, N*sizeof(double));
  cudaMalloc(&d_dv_3_z, N*sizeof(double));

  cudaMalloc(&d_dv_1pr_x, N*sizeof(double));
  cudaMalloc(&d_dv_1pr_y, N*sizeof(double));
  cudaMalloc(&d_dv_1pr_z, N*sizeof(double));

  cudaMalloc(&d_dv_2pr_x, N*sizeof(double));
  cudaMalloc(&d_dv_2pr_y, N*sizeof(double));
  cudaMalloc(&d_dv_2pr_z, N*sizeof(double));

  cudaMalloc(&d_dv_3pr_x, N*sizeof(double));
  cudaMalloc(&d_dv_3pr_y, N*sizeof(double));
  cudaMalloc(&d_dv_3pr_z, N*sizeof(double));

  cudaMalloc(&d_v1_x, N*sizeof(double));
  cudaMalloc(&d_v1_y, N*sizeof(double));
  cudaMalloc(&d_v1_z, N*sizeof(double));

  cudaMalloc(&d_v2_x, N*sizeof(double));
  cudaMalloc(&d_v2_y, N*sizeof(double));
  cudaMalloc(&d_v2_z, N*sizeof(double));

  cudaMalloc(&d_v3_x, N*sizeof(double));
  cudaMalloc(&d_v3_y, N*sizeof(double));
  cudaMalloc(&d_v3_z, N*sizeof(double));

  cudaMalloc(&d_v1_prime_x, N*sizeof(double));
  cudaMalloc(&d_v1_prime_y, N*sizeof(double));
  cudaMalloc(&d_v1_prime_z, N*sizeof(double));

  cudaMalloc(&d_v2_prime_x, N*sizeof(double));
  cudaMalloc(&d_v2_prime_y, N*sizeof(double));
  cudaMalloc(&d_v2_prime_z, N*sizeof(double));

  cudaMalloc(&d_v3_prime_x, N*sizeof(double));
  cudaMalloc(&d_v3_prime_y, N*sizeof(double));
  cudaMalloc(&d_v3_prime_z, N*sizeof(double));

  cudaMalloc(&d_nv1_x, N*sizeof(double));
  cudaMalloc(&d_nv1_y, N*sizeof(double));
  cudaMalloc(&d_nv1_z, N*sizeof(double));

  cudaMalloc(&d_nv2_x, N*sizeof(double));
  cudaMalloc(&d_nv2_y, N*sizeof(double));
  cudaMalloc(&d_nv2_z, N*sizeof(double));

  cudaMalloc(&d_nv3_x, N*sizeof(double));
  cudaMalloc(&d_nv3_y, N*sizeof(double));
  cudaMalloc(&d_nv3_z, N*sizeof(double));

  cudaMalloc(&d_nv1_prime_x, N*sizeof(double));
  cudaMalloc(&d_nv1_prime_y, N*sizeof(double));
  cudaMalloc(&d_nv1_prime_z, N*sizeof(double));

  cudaMalloc(&d_nv2_prime_x, N*sizeof(double));
  cudaMalloc(&d_nv2_prime_y, N*sizeof(double));
  cudaMalloc(&d_nv2_prime_z, N*sizeof(double));

  cudaMalloc(&d_nv3_prime_x, N*sizeof(double));
  cudaMalloc(&d_nv3_prime_y, N*sizeof(double));
  cudaMalloc(&d_nv3_prime_z, N*sizeof(double));

  cudaMalloc(&d_still_together, N*sizeof(bool));
  cudaMalloc(&d_times, N*sizeof(int));
  cudaMalloc(&d_not_diverged, N*sizeof(bool));

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

  cudaMemcpy(d_p1_x, p1_x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_p1_y, p1_y, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_p1_z, p1_z, N*sizeof(double), cudaMemcpyHostToDevice);

  cudaMemcpy(d_p2_x, p2_x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_p2_y, p2_y, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_p2_z, p2_z, N*sizeof(double), cudaMemcpyHostToDevice);

  cudaMemcpy(d_p3_x, p3_x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_p3_y, p3_y, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_p3_z, p3_z, N*sizeof(double), cudaMemcpyHostToDevice);

  cudaMemcpy(d_p1_prime_x, p1_prime_x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_p1_prime_y, p1_prime_y, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_p1_prime_z, p1_prime_z, N*sizeof(double), cudaMemcpyHostToDevice);

  cudaMemcpy(d_p2_prime_x, p2_prime_x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_p2_prime_y, p2_prime_y, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_p2_prime_z, p2_prime_z, N*sizeof(double), cudaMemcpyHostToDevice);

  cudaMemcpy(d_p3_prime_x, p3_prime_x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_p3_prime_y, p3_prime_y, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_p3_prime_z, p3_prime_z, N*sizeof(double), cudaMemcpyHostToDevice);

  cudaMemcpy(d_dv_1_x, dv_1_x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dv_1_y, dv_1_y, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dv_1_z, dv_1_z, N*sizeof(double), cudaMemcpyHostToDevice);

  cudaMemcpy(d_dv_2_x, dv_2_x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dv_2_y, dv_2_y, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dv_2_z, dv_2_z, N*sizeof(double), cudaMemcpyHostToDevice);

  cudaMemcpy(d_dv_3_x, dv_3_x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dv_3_y, dv_3_y, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dv_3_z, dv_3_z, N*sizeof(double), cudaMemcpyHostToDevice);

  cudaMemcpy(d_dv_1pr_x, dv_1pr_x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dv_1pr_y, dv_1pr_y, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dv_1pr_z, dv_1pr_z, N*sizeof(double), cudaMemcpyHostToDevice);

  cudaMemcpy(d_dv_2pr_x, dv_2pr_x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dv_2pr_y, dv_2pr_y, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dv_2pr_z, dv_2pr_z, N*sizeof(double), cudaMemcpyHostToDevice);

  cudaMemcpy(d_dv_3pr_x, dv_3pr_x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dv_3pr_y, dv_3pr_y, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dv_3pr_z, dv_3pr_z, N*sizeof(double), cudaMemcpyHostToDevice);

  cudaMemcpy(d_v1_x, v1_x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v1_y, v1_y, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v1_z, v1_z, N*sizeof(double), cudaMemcpyHostToDevice);

  cudaMemcpy(d_v2_x, v2_x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v2_y, v2_y, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v2_z, v2_z, N*sizeof(double), cudaMemcpyHostToDevice);

  cudaMemcpy(d_v3_x, v3_x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v3_y, v3_y, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v3_z, v3_z, N*sizeof(double), cudaMemcpyHostToDevice);

  cudaMemcpy(d_v1_prime_x, v1_prime_x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v1_prime_y, v1_prime_y, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v1_prime_z, v1_prime_z, N*sizeof(double), cudaMemcpyHostToDevice);

  cudaMemcpy(d_v2_prime_x, v2_prime_x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v2_prime_y, v2_prime_y, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v2_prime_z, v2_prime_z, N*sizeof(double), cudaMemcpyHostToDevice);

  cudaMemcpy(d_v3_prime_x, v3_prime_x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v3_prime_y, v3_prime_y, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v3_prime_z, v3_prime_z, N*sizeof(double), cudaMemcpyHostToDevice);

  cudaMemcpy(d_nv1_x, nv1_x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_nv1_y, nv1_y, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_nv1_z, nv1_z, N*sizeof(double), cudaMemcpyHostToDevice);

  cudaMemcpy(d_nv2_x, nv2_x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_nv2_y, nv2_y, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_nv2_z, nv2_z, N*sizeof(double), cudaMemcpyHostToDevice);

  cudaMemcpy(d_nv3_x, nv3_x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_nv3_y, nv3_y, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_nv3_z, nv3_z, N*sizeof(double), cudaMemcpyHostToDevice);

  cudaMemcpy(d_nv1_prime_x, nv1_prime_x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_nv1_prime_y, nv1_prime_y, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_nv1_prime_z, nv1_prime_z, N*sizeof(double), cudaMemcpyHostToDevice);

  cudaMemcpy(d_nv2_prime_x, nv2_prime_x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_nv2_prime_y, nv2_prime_y, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_nv2_prime_z, nv2_prime_z, N*sizeof(double), cudaMemcpyHostToDevice);

  cudaMemcpy(d_nv3_prime_x, nv3_prime_x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_nv3_prime_y, nv3_prime_y, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_nv3_prime_z, nv3_prime_z, N*sizeof(double), cudaMemcpyHostToDevice);

  cudaMemcpy(d_times, times, N*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_still_together, still_together, N*sizeof(bool), cudaMemcpyHostToDevice);
  cudaMemcpy(d_not_diverged, not_diverged, N*sizeof(bool), cudaMemcpyHostToDevice);

  // initialize timer
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();

  // call CUDA kernal on inputs in configuration <<< blockIdx, threadIdx>>>>
  divergence<<<(N+127)/128, 128>>>(
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
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  std::cout << "Elapsed Time: " << elapsed_seconds.count() << "s\n";

  cudaMemcpy(times, d_times, N*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(still_together, d_still_together, N*sizeof(bool), cudaMemcpyDeviceToHost);
  cudaMemcpy(not_diverged, d_not_diverged, N*sizeof(bool), cudaMemcpyDeviceToHost);
  cudaMemcpy(p1_x, d_p1_x, N*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(p1_y, d_p1_y, N*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(p1_z, d_p1_z, N*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(p1_prime_x, d_p1_prime_x, N*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(p1_prime_y, d_p1_prime_y, N*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(p1_prime_z, d_p1_prime_z, N*sizeof(double), cudaMemcpyDeviceToHost);

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

  cudaFree(d_v1_x); cudaFree(d_v1_y); cudaFree(d_v1_z);
  cudaFree(d_v2_x); cudaFree(d_v2_y); cudaFree(d_v2_z);
  cudaFree(d_v3_x); cudaFree(d_v3_y); cudaFree(d_v3_z);

  cudaFree(d_v1_prime_x); cudaFree(d_v1_prime_y); cudaFree(d_v1_prime_z);
  cudaFree(d_v2_prime_x); cudaFree(d_v2_prime_y); cudaFree(d_v2_prime_z);
  cudaFree(d_v3_prime_x); cudaFree(d_v3_prime_y); cudaFree(d_v3_prime_z);

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

  free(v1_x); free(v1_y); free(v1_z);
  free(v2_x); free(v2_y); free(v2_z);
  free(v3_x); free(v3_y); free(v3_z);

  free(v1_prime_x); free(v1_prime_y); free(v1_prime_z);
  free(v2_prime_x); free(v2_prime_y); free(v2_prime_z);
  free(v3_prime_x); free(v3_prime_y); free(v3_prime_z);

  free(nv1_x); free(nv1_y); free(nv1_z);
  free(nv2_x); free(nv2_y); free(nv2_z);
  free(nv3_x); free(nv3_y); free(nv3_z);

  free(nv1_prime_x); free(nv1_prime_y); free(nv1_prime_z);
  free(nv2_prime_x); free(nv2_prime_y); free(nv2_prime_z);
  free(nv3_prime_x); free(nv3_prime_y); free(nv3_prime_z);


  free(times); free(still_together); free(not_diverged);

}















