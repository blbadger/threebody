#include <stdio.h>
#include <iostream>

__global__
void divergence(int n, 
              int steps,
              float delta_t,
              int *still_together,
              bool *not_diverged,
              int *times,
              float m_1,
              float m_2, 
              float m_3,
              float critical_distance,
              float *p1_x, float *p1_y, float *p1_z, 
              float *p2_x, float *p2_y, float *p2_z, 
              float *p3_x, float *p3_y, float *p3_x, 
              float *p1_prime_x, float *p1_prime_y, float *p1_prime_z, 
              float *p2_prime_x, float *p2_prime_y, float *p2_prime_z, 
              float *p3_prime_x, float *p3_prime_y, float *p3_prime_z, 
              float *dv_1_x, float *dv_1_y, float *dv_1_z,
              float *dv_2_x, float *dv_2_y, float *dv_2_z,
              float *dv_3_x, float *dv_3_y, float *dv_3_z,
              float *dv_1pr_x, float *dv_1pr_y, float *dv_1pr_z,
              float *dv_2pr_x, float *dv_2pr_y, float *dv_2pr_z,
              float *dv_3pr_x, float *dv_3pr_y, float *dv_3pr_z,
              float *nv1_x, float *nv1_y, float *nv1_z,
              float *nv2_x, float *nv2_y, float *nv2_z,
              float *nv3_x, float *nv3_y, float *nv3_z,
              float *nv1_prime_x, float *nv1_prime_y, float *nv1_prime_z,
              float *nv2_prime_x, float *nv2_prime_y, float *nv2_prime_z,
              float *nv3_prime_x, float *nv3_prime_y, float *nv3_prime_z,
              )
              
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  for (int j=0; j < steps; j++) {
    if (i < n) {
      // compute accelerations
      dv_1_x[i] = -9.8f * m_2 * (p1_x[i] - p2_x[i]) / powf(sqrt(powf(p1_x[i] - p2_x[i]), 2) + powf(p1_y[i] - p2_y[i], 2) + powf(p1_z[i] - p2_z[i], 2), 3) \
                  -9.8f * m_3 * (p1_x[i] - p3_x[i]) / powf(sqrt(powf(p1_x[i] - p3_x[i]), 2) + powf(p1_y[i] - p3_y[i], 2) + powf(p1_z[i] - p3_z[i], 2), 3);

      dv_1_y[i] = -9.8f * m_2 * (p1_y[i] - p2_y[i]) / powf(sqrt(powf(p1_x[i] - p2_x[i]), 2) + powf(p1_y[i] - p2_y[i], 2) + powf(p1_z[i] - p2_z[i], 2), 3) \
                  -9.8f * m_3 * (p1_y[i] - p3_y[i]) / powf(sqrt(powf(p1_x[i] - p3_x[i]), 2) + powf(p1_y[i] - p3_y[i], 2) + powf(p1_z[i] - p3_z[i], 2), 3);

      dv_1_z[i] = -9.8f * m_2 * (p1_z[i] - p2_z[i]) / powf(sqrt(powf(p1_x[i] - p2_x[i]), 2) + powf(p1_y[i] - p2_y[i], 2) + powf(p1_z[i] - p2_z[i], 2), 3) \
                  -9.8f * m_3 * (p1_z[i] - p3_z[i]) / powf(sqrt(powf(p1_x[i] - p3_x[i]), 2) + powf(p1_y[i] - p3_y[i], 2) + powf(p1_z[i] - p3_z[i], 2), 3);

      dv_2_x[i] = -9.8f * m_3 * (p1_x[i] - p3_x[i]) / powf(sqrt(powf(p2_x[i] - p3_x[i]), 2) + powf(p2_y[i] - p3_y[i], 2) + powf(p2_z[i] - p3_z[i], 2), 3) \
                  -9.8f * m_1 * (p1_x[i] - p3_x[i]) / powf(sqrt(powf(p2_x[i] - p1_x[i]), 2) + powf(p2_y[i] - p1_y[i], 2) + powf(p2_z[i] - p1_z[i], 2), 3);

      dv_2_y[i] = -9.8f * m_3 * (p1_y[i] - p3_y[i]) / powf(sqrt(powf(p2_x[i] - p3_x[i]), 2) + powf(p2_y[i] - p3_y[i], 2) + powf(p2_z[i] - p3_z[i], 2), 3) \
                  -9.8f * m_1 * (p1_y[i] - p3_y[i]) / powf(sqrt(powf(p2_x[i] - p1_x[i]), 2) + powf(p2_y[i] - p1_y[i], 2) + powf(p2_z[i] - p1_z[i], 2), 3);

      dv_2_z[i] = -9.8f * m_3 * (p1_z[i] - p3_z[i]) / powf(sqrt(powf(p2_x[i] - p3_x[i]), 2) + powf(p2_y[i] - p3_y[i], 2) + powf(p2_z[i] - p3_z[i], 2), 3) \
                  -9.8f * m_1 * (p1_z[i] - p3_z[i]) / powf(sqrt(powf(p2_x[i] - p1_x[i]), 2) + powf(p2_y[i] - p1_y[i], 2) + powf(p2_z[i] - p1_z[i], 2), 3);

      dv_3_x[i] = -9.8f * m_1 * (p1_x[i] - p3_x[i]) / powf(sqrt(powf(p3_x[i] - p1_x[i]), 2) + powf(p3_y[i] - p1_y[i], 2) + powf(p3_z[i] - p1_z[i], 2), 3) \
                  -9.8f * m_2 * (p1_x[i] - p3_x[i]) / powf(sqrt(powf(p3_x[i] - p2_x[i]), 2) + powf(p3_y[i] - p2_y[i], 2) + powf(p3_z[i] - p2_z[i], 2), 3);

      dv_3_y[i] = -9.8f * m_1 * (p1_y[i] - p3_y[i]) / powf(sqrt(powf(p3_x[i] - p1_x[i]), 2) + powf(p3_y[i] - p1_y[i], 2) + powf(p3_z[i] - p1_z[i], 2), 3) \
                  -9.8f * m_2 * (p1_y[i] - p3_y[i]) / powf(sqrt(powf(p3_x[i] - p2_x[i]), 2) + powf(p3_y[i] - p2_y[i], 2) + powf(p3_z[i] - p2_z[i], 2), 3);

      dv_3_z[i] = -9.8f * m_1 * (p1_z[i] - p3_z[i]) / powf(sqrt(powf(p3_x[i] - p1_x[i]), 2) + powf(p3_y[i] - p1_y[i], 2) + powf(p3_z[i] - p1_z[i], 2), 3) \
                  -9.8f * m_2 * (p1_z[i] - p3_z[i]) / powf(sqrt(powf(p3_x[i] - p2_x[i]), 2) + powf(p3_y[i] - p2_y[i], 2) + powf(p3_z[i] - p2_z[i], 2), 3);

      dv_1pr_x[i] = -9.8f * m_2 * (p1_prime_x[i] - p2_prime_x[i]) / powf(sqrt(powf(p1_prime_x[i] - p2_prime_x[i]), 2) + powf(p1_prime_y[i] - p1_prime_y[i], 2) + powf(p1_prime_z[i] - p2_prime_z[i], 2), 3) \
                    -9.8f * m_3 * (p1_prime_x[i] - p3_prime_x[i]) / powf(sqrt(powf(p1_prime_x[i] - p3_prime_x[i]), 2) + powf(p1_prime_y[i] - p3_prime_y[i], 2) + powf(p1_prime_z[i] - p3_prime_z[i], 2), 3);

      dv_1pr_y[i] = -9.8f * m_2 * (p1_prime_y[i] - p1_prime_y[i]) / powf(sqrt(powf(p1_prime_x[i] - p2_prime_x[i]), 2) + powf(p1_prime_y[i] - p1_prime_y[i], 2) + powf(p1_prime_z[i] - p2_prime_z[i], 2), 3) \
                    -9.8f * m_3 * (p1_prime_y[i] - p3_prime_y[i]) / powf(sqrt(powf(p1_prime_x[i] - p3_prime_x[i]), 2) + powf(p1_prime_y[i] - p3_prime_y[i], 2) + powf(p1_prime_z[i] - p3_prime_z[i], 2), 3);

      dv_1pr_z[i] = -9.8f * m_2 * (p1_prime_z[i] - p2_prime_z[i]) / powf(sqrt(powf(p1_prime_x[i] - p2_prime_x[i]), 2) + powf(p1_prime_y[i] - p1_prime_y[i], 2) + powf(p1_prime_z[i] - p2_prime_z[i], 2), 3) \
                    -9.8f * m_3 * (p1_prime_z[i] - p3_prime_z[i]) / powf(sqrt(powf(p1_prime_x[i] - p3_prime_x[i]), 2) + powf(p1_prime_y[i] - p3_prime_y[i], 2) + powf(p1_prime_z[i] - p3_prime_z[i], 2), 3);

      dv_2pr_x[i] = -9.8f * m_3 * (p1_prime_x[i] - p3_prime_x[i]) / powf(sqrt(powf(p2_prime_x[i] - p3_prime_x[i]), 2) + powf(p1_prime_y[i] - p3_prime_y[i], 2) + powf(p2_prime_z[i] - p3_prime_z[i], 2), 3) \
                    -9.8f * m_1 * (p1_prime_x[i] - p3_prime_x[i]) / powf(sqrt(powf(p2_prime_x[i] - p1_prime_x[i]), 2) + powf(p1_prime_y[i] - p1_prime_y[i], 2) + powf(p2_prime_z[i] - p1_prime_z[i], 2), 3);

      dv_2pr_y[i] = -9.8f * m_3 * (p1_prime_y[i] - p3_prime_y[i]) / powf(sqrt(powf(p2_prime_x[i] - p3_prime_x[i]), 2) + powf(p1_prime_y[i] - p3_prime_y[i], 2) + powf(p2_prime_z[i] - p3_prime_z[i], 2), 3) \
                    -9.8f * m_1 * (p1_prime_y[i] - p3_prime_y[i]) / powf(sqrt(powf(p2_prime_x[i] - p1_prime_x[i]), 2) + powf(p1_prime_y[i] - p1_prime_y[i], 2) + powf(p2_prime_z[i] - p1_prime_z[i], 2), 3);

      dv_2pr_z[i] = -9.8f * m_3 * (p1_prime_z[i] - p3_prime_z[i]) / powf(sqrt(powf(p2_prime_x[i] - p3_prime_x[i]), 2) + powf(p1_prime_y[i] - p3_prime_y[i], 2) + powf(p2_prime_z[i] - p3_prime_z[i], 2), 3) \
                    -9.8f * m_1 * (p1_prime_z[i] - p3_prime_z[i]) / powf(sqrt(powf(p2_prime_x[i] - p1_prime_x[i]), 2) + powf(p1_prime_y[i] - p1_prime_y[i], 2) + powf(p2_prime_z[i] - p1_prime_z[i], 2), 3);

      dv_3pr_x[i] = -9.8f * m_1 * (p1_prime_x[i] - p3_prime_x[i]) / powf(sqrt(powf(p3_prime_x[i] - p1_prime_x[i]), 2) + powf(p3_prime_y[i] - p1_prime_y[i], 2) + powf(p3_prime_z[i] - p1_prime_z[i], 2), 3) \
                    -9.8f * m_2 * (p1_prime_x[i] - p3_prime_x[i]) / powf(sqrt(powf(p3_prime_x[i] - p2_prime_x[i]), 2) + powf(p3_prime_y[i] - p1_prime_y[i], 2) + powf(p3_prime_z[i] - p2_prime_z[i], 2), 3);

      dv_3pr_y[i] = -9.8f * m_1 * (p1_prime_y[i] - p3_prime_y[i]) / powf(sqrt(powf(p3_prime_x[i] - p1_prime_x[i]), 2) + powf(p3_prime_y[i] - p1_prime_y[i], 2) + powf(p3_prime_z[i] - p1_prime_z[i], 2), 3) \
                    -9.8f * m_2 * (p1_prime_y[i] - p3_prime_y[i]) / powf(sqrt(powf(p3_prime_x[i] - p2_prime_x[i]), 2) + powf(p3_prime_y[i] - p1_prime_y[i], 2) + powf(p3_prime_z[i] - p2_prime_z[i], 2), 3);

      dv_3pr_z[i] = -9.8f * m_1 * (p1_prime_z[i] - p3_prime_z[i]) / powf(sqrt(powf(p3_prime_x[i] - p1_prime_x[i]), 2) + powf(p3_prime_y[i] - p1_prime_y[i], 2) + powf(p3_prime_z[i] - p1_prime_z[i], 2), 3) \
                    -9.8f * m_2 * (p1_prime_z[i] - p3_prime_z[i]) / powf(sqrt(powf(p3_prime_x[i] - p2_prime_x[i]), 2) + powf(p3_prime_y[i] - p1_prime_y[i], 2) + powf(p3_prime_z[i] - p2_prime_z[i], 2), 3);

      // find which trajectories have diverged and save to *diverged
      not_diverged[i] = sqrt(powf(p1_x[i] - p1_prime_x[i], 2) + powf(p1_y[i] - p1_prime_y[i], 2) + powf(p1_z[i] - p1_prime_z[i], 2)) <= critical_distance;
      still_together[i] &= not_diverged[i];
      if (still_together[i]){
        times[i]++
      }

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
  std::cout << N;
  int steps = 50000;
  float *p1_x, *p1_y, *p1_z;
  float *p2_x, *p2_y, *p2_z;
  float *p3_x, *p3_y, *p3_z;
  float *p1_prime_x, *p1_prime_y, *p1_prime_z;
  float *p2_prime_x, *p2_prime_y, *p2_prime_z;
  float *p3_prime_x, *p3_prime_y, *p3_prime_z;
  float *dv_1_x, *dv_1_y, *dv_1_z;
  float *dv_2_x, *dv_2_y, *dv_2_z;
  float *dv_3_x, *dv_3_y, *dv_3_z;

  float *d_p1_x, *d_p1_y, *d_p1_z;
  float *d_p2_x, *d_p2_y, *d_p2_z;
  float *d_p3_x, *d_p3_y, *d_p3_z;
  float *d_p1_prime_x, *d_p1_prime_y, *d_p1_prime_z;
  float *d_p2_prime_x, *d_p2_prime_y, *d_p2_prime_z;
  float *d_p3_prime_x, *d_p3_prime_y, *d_p3_prime_z;
  float *dv_1pr_x, *dv_1pr_y, *dv_1pr_z;
  float *dv_2pr_x, *dv_2pr_y, *dv_2pr_z;
  float *dv_3pr_x, *dv_3pr_y, *dv_3pr_z;

  int *diverged, *d_diverged;

  p1_x = (float*)malloc(N*sizeof(float));
  p1_y = (float*)malloc(N*sizeof(float));
  p1_z = (float*)malloc(N*sizeof(float));

  p2_x = (float*)malloc(N*sizeof(float));
  p2_y = (float*)malloc(N*sizeof(float));
  p2_z = (float*)malloc(N*sizeof(float));

  p3_x = (float*)malloc(N*sizeof(float));
  p3_y = (float*)malloc(N*sizeof(float));
  p3_z = (float*)malloc(N*sizeof(float));

  p1_prime_x = (float*)malloc(N*sizeof(float));
  p1_prime_y = (float*)malloc(N*sizeof(float));
  p1_prime_z = (float*)malloc(N*sizeof(float));

  p2_prime_x = (float*)malloc(N*sizeof(float));
  p2_prime_y = (float*)malloc(N*sizeof(float));
  p2_prime_z = (float*)malloc(N*sizeof(float));

  p3_prime_x = (float*)malloc(N*sizeof(float));
  p3_prime_y = (float*)malloc(N*sizeof(float));
  p3_prime_z = (float*)malloc(N*sizeof(float));

  dv_1_x = (float*)malloc(N*sizeof(int));
  dv_1_y = (float*)malloc(N*sizeof(int));
  dv_1_z = (float*)malloc(N*sizeof(int));

  dv_2_x = (float*)malloc(N*sizeof(int));
  dv_2_y = (float*)malloc(N*sizeof(int));
  dv_2_z = (float*)malloc(N*sizeof(int));

  dv_3_x = (float*)malloc(N*sizeof(int));
  dv_3_y = (float*)malloc(N*sizeof(int));
  dv_3_z = (float*)malloc(N*sizeof(int));

  dv_1pr_x = (float*)malloc(N*sizeof(int));
  dv_1pr_y = (float*)malloc(N*sizeof(int));
  dv_1pr_z = (float*)malloc(N*sizeof(int));

  dv_2pr_x = (float*)malloc(N*sizeof(int));
  dv_2pr_y = (float*)malloc(N*sizeof(int));
  dv_2pr_z = (float*)malloc(N*sizeof(int));

  dv_3pr_x = (float*)malloc(N*sizeof(int));
  dv_3pr_y = (float*)malloc(N*sizeof(int));
  dv_3pr_z = (float*)malloc(N*sizeof(int));


  cudaMalloc(&d_p1_x, N*sizeof(float)); 
  cudaMalloc(&d_p1_y, N*sizeof(float)); 
  cudaMalloc(&d_p1_z, N*sizeof(float)); 

  cudaMalloc(&d_p2_x, N*sizeof(float));
  cudaMalloc(&d_p2_y, N*sizeof(float));
  cudaMalloc(&d_p2_z, N*sizeof(float));

  cudaMalloc(&d_p3_x, N*sizeof(float));
  cudaMalloc(&d_p3_y, N*sizeof(float));
  cudaMalloc(&d_p3_z, N*sizeof(float));

  cudaMalloc(&d_p1_prime, N*sizeof(float));
  cudaMalloc(&d_p2_prime, N*sizeof(float));
  cudaMalloc(&d_p3_prime, N*sizeof(float));

  cudaMalloc(&d_p_dv_1_x, N*sizeof(float));
  cudaMalloc(&d_p_dv_1_x, N*sizeof(float));
  cudaMalloc(&d_p_dv_1_x, N*sizeof(float));

  cudaMalloc(&d_p_dv_2_x, N*sizeof(float));
  cudaMalloc(&d_p_dv_2_y, N*sizeof(float));
  cudaMalloc(&d_p_dv_2_z, N*sizeof(float));

  cudaMalloc(&d_p_dv_3_x, N*sizeof(float));
  cudaMalloc(&d_p_dv_3_y, N*sizeof(float));
  cudaMalloc(&d_p_dv_3_z, N*sizeof(float));

  cudaMalloc(&d_diverged, N*sizeof(int));

  for (int i = 0; i < N; i++) {
    p1_x[i] = 0.0f + i;
    p1_y[i] = 0.0f + i;
    p1_z[i] = 0.0f + i;

    p2_x[i] = 0.0f + i;
    p2_y[i] = 0.0f + i;
    p2_z[i] = 0.0f + i;

    p3_x[i] = 0.0f + i;
    p3_y[i] = 0.0f + i;
    p3_z[i] = 0.0f + i;

    p1_prime[i] = 0.0f + i + 0.0001f;
    p2_prime[i] = 0.0f + i;
    p3_prime[i] = 0.0f + i;
    diverged[i] = 0;

    p_dv_1_x[i] = 0;
    p_dv_1_x[i] = 0;
    p_dv_1_x[i] = 0;

    p_dv_2[i] = 0;
    p_dv_2[i] = 0;
    p_dv_2[i] = 0;

    p_dv_3[i] = 0;
    p_dv_3[i] = 0;
    p_dv_3[i] = 0;

  }

  cudaMemcpy(d_p1_x p1_x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_p1_y, p1_y, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_p1_z, p1_z, N*sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(d_p2_x, p2_x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_p2_y, p2_y, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_p2_z, p2_z, N*sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(d_p3_x, p3_x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_p3_y, p3_y, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_p3_z, p3_z, N*sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(d_p1_prime, p1_prime, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_p2_prime, p2_prime, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_p3_prime, p3_prime, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_diverged, diverged, N*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_p_dv_1, p_dv_1, N*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_p_dv_2, p_dv_2, N*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_p_dv_3, p_dv_3, N*sizeof(int), cudaMemcpyHostToDevice);

  cudaMemcpy(d_times, times, N*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_still_together, still_together, N*sizeof(bool), cudaMemcpyHostToDevice);



  // call CUDA kernal
  divergence<<<(N+255)/256, 256>>>(N, 
    steps, 
    d_p1_x, d_p1_y, d_p1_z, 
    d_p2_x, d_p2_y, d_p2_z, 
    d_p3_x, d_p3_y, d_p3_z, 
    d_p1_prime, 
    d_p2_prime, 
    d_p3_prime, 
    d_diverged, 
    d_p_dv_1, d_p_dv_2, d_p_dv_3);

  cudaMemcpy(diverged, d_diverged, N*sizeof(float), cudaMemcpyDeviceToHost);
  for (int i=0; i < 100; i++) {
    std::cout<< diverged[i];
    std::cout << ' ';
  }

  cudaFree(d_p1);
  cudaFree(d_p3);
  cudaFree(d_p3);
  cudaFree(d_p1_prime);
  cudaFree(d_p2_prime);
  cudaFree(d_p3_prime);

  free(p1_x); free(p1_y); free(p1_z);
  free(p2_y);
  free(p3_z);
  free(p1_prime);
  free(p2_prime);
  free(p3_prime);
}













