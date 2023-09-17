#include <stdio.h>
#include <iostream>

__global__
void divergence(int n, 
              int steps,
              float delta_t,
              float m_1,
              float m_2, 
              float m_3,
              float critical_distance,
              float *p1_x, float *p1_y, float *p1_z, 
              float *p2_x, float *p2_y, float *p2_z, 
              float *p3_x, float *p2_y, float *p2_x, 
              float *p1_prime, 
              float *p2_prime, 
              float *p3_prime, 
              int *diverged,
              float *p_dv_1_x, float *p_dv_1_y, float *p_dv_1_z,
              float *p_dv_2_x, float *p_dv_2_y, float *p_dv_2_z, 
              float *p_dv_3_x, float *p_dv_3_y, float *p_dv_3_z)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  for (int j=0; j < steps; j++) {
    if (i < n) {
      p_dv_1_x[i] = -9.8f * m_2 * (p1_x[i] - p2_x[i]) / powf(sqrt(powf(p1_x[i] - p2_x[i]), 2) + powf(p1_y[i] - p2_y[i], 2) + powf(p1_z[i] - p2_z[i], 2), 3) \
                    -9.8f * m_3 * (p1_x[i] - p3_x[i]) / powf(sqrt(powf(p1_x[i] - p3_x[i]), 2) + powf(p1_y[i] - p3_y[i], 2) + powf(p1_z[i] - p3_z[i], 2), 3);

      p_dv_1_y[i] = -9.8f * m_2 * (p1_y[i] - p2_y[i]) / powf(sqrt(powf(p1_x[i] - p2_x[i]), 2) + powf(p1_y[i] - p2_y[i], 2) + powf(p1_z[i] - p2_z[i], 2), 3) \
                    -9.8f * m_3 * (p1_y[i] - p3_y[i]) / powf(sqrt(powf(p1_x[i] - p3_x[i]), 2) + powf(p1_y[i] - p3_y[i], 2) + powf(p1_z[i] - p3_z[i], 2), 3);

      p_dv_1_z[i] = -9.8f * m_2 * (p1_z[i] - p2_z[i]) / powf(sqrt(powf(p1_x[i] - p2_x[i]), 2) + powf(p1_y[i] - p2_y[i], 2) + powf(p1_z[i] - p2_z[i], 2), 3) \
                    -9.8f * m_3 * (p1_z[i] - p3_z[i]) / powf(sqrt(powf(p1_x[i] - p3_x[i]), 2) + powf(p1_y[i] - p3_y[i], 2) + powf(p1_z[i] - p3_z[i], 2), 3);


      p_dv_2_x[i] = -9.8f * m_3 * (p1[i] - p3[i]) / powf(sqrt(powf(p2_x[i] - p3_x[i]), 2) + powf(p2_y[i] - p3_y[i], 2) + powf(p2_z[i] - p3_z[i], 2), 3) \
                    -9.8f * m_1 * (p1[i] - p3[i]) / powf(sqrt(powf(p2_x[i] - p1_x[i]), 2) + powf(p2_y[i] - p1_y[i], 2) + powf(p2_z[i] - p1_z[i], 2), 3);

      p_dv_2_y[i] = -9.8f * m_3 * (p1[i] - p3[i]) / powf(sqrt(powf(p2_x[i] - p3_x[i]), 2) + powf(p2_y[i] - p3_y[i], 2) + powf(p2_z[i] - p3_z[i], 2), 3) \
                    -9.8f * m_1 * (p1[i] - p3[i]) / powf(sqrt(powf(p2_x[i] - p1_x[i]), 2) + powf(p2_y[i] - p1_y[i], 2) + powf(p2_z[i] - p1_z[i], 2), 3);

      p_dv_2_z[i] = -9.8f * m_3 * (p1[i] - p3[i]) / powf(sqrt(powf(p2_x[i] - p3_x[i]), 2) + powf(p2_y[i] - p3_y[i], 2) + powf(p2_z[i] - p3_z[i], 2), 3) \
                    -9.8f * m_1 * (p1[i] - p3[i]) / powf(sqrt(powf(p2_x[i] - p1_x[i]), 2) + powf(p2_y[i] - p1_y[i], 2) + powf(p2_z[i] - p1_z[i], 2), 3);


      p_dv_3_x[i] = -9.8f * m_1 * (p1_z[i] - p3_z[i]) / powf(sqrt(powf(p3_x[i] - p1_x[i]), 2) + powf(p3_y[i] - p1_y[i], 2) + powf(p3_z[i] - p1_z[i], 2), 3) \
                    -9.8f * m_2 * (p1_z[i] - p3_z[i]) / powf(sqrt(powf(p3_x[i] - p2_x[i]), 2) + powf(p3_y[i] - p2_y[i], 2) + powf(p3_z[i] - p2_z[i], 2), 3);

      p_dv_3_y[i] = -9.8f * m_1 * (p1_y[i] - p3_y[i]) / powf(sqrt(powf(p3_x[i] - p1_x[i]), 2) + powf(p3_y[i] - p1_y[i], 2) + powf(p3_z[i] - p1_z[i], 2), 3) \
                    -9.8f * m_2 * (p1_y[i] - p3_y[i]) / powf(sqrt(powf(p3_x[i] - p2_x[i]), 2) + powf(p3_y[i] - p2_y[i], 2) + powf(p3_z[i] - p2_z[i], 2), 3);

      p_dv_3_z[i] = -9.8f * m_1 * (p1_z[i] - p3_z[i]) / powf(sqrt(powf(p3_x[i] - p1_x[i]), 2) + powf(p3_y[i] - p1_y[i], 2) + powf(p3_z[i] - p1_z[i], 2), 3) \
                    -9.8f * m_2 * (p1_z[i] - p3_z[i]) / powf(sqrt(powf(p3_x[i] - p2_x[i]), 2) + powf(p3_y[i] - p2_y[i], 2) + powf(p3_z[i] - p2_z[i], 2), 3);


      p1_x[i] = p1_x[i] + delta_t * p_dv_1_x[i];
      p1_y[i] = p1_y[i] + delta_t * p_dv_1_y[i];
      p1_z[i] = p1_z[i] + delta_t * p_dv_1_z[i];

      p2_x[i] = p2_x[i] + delta_t * p_dv_2_x[i];
      p2_y[i] = p2_y[i] + delta_t * p_dv_2_y[i];
      p2_z[i] = p2_z[i] + delta_t * p_dv_2_z[i];

      p3_x[i] = p3_x[i] + delta_t * p_dv_3_x[i];
      p3_y[i] = p3_y[i] + delta_t * p_dv_3_y[i];
      p3_z[i] = p3_z[i] + delta_t * p_dv_3_z[i];

      // find which trajectories have diverged and save to *diverged
    }
  }
}


int main(void)
{
  int N = 1000;
  std::cout << N;
  int steps = 50000;
  float *p1_x, *p1_y, *p1_z;
  float *p2_x, *p2_y, *p2_z;
  float *p3_x, *p3_y, *p3_z;
  float *p1_prime, *p2_prime, *p3_prime, *d_p1, *d_p2, *d_p3, *d_p1_prime, *d_p2_prime, *d_p3_prime;
  int *diverged, *d_diverged;
  float *p_dv_1, *p_dv_2, *p_dv_3, *d_p_dv_1, *d_p_dv_2, *d_p_dv_3;
  p1_x = (float*)malloc(N*sizeof(float));
  p1_y = (float*)malloc(N*sizeof(float));
  p1_z = (float*)malloc(N*sizeof(float));

  p2_x = (float*)malloc(N*sizeof(float));
  p2_y = (float*)malloc(N*sizeof(float));
  p2_z = (float*)malloc(N*sizeof(float));

  p3_x = (float*)malloc(N*sizeof(float));
  p3_y = (float*)malloc(N*sizeof(float));
  p3_z = (float*)malloc(N*sizeof(float));

  p1_prime = (float*)malloc(N*sizeof(float));
  p2_prime = (float*)malloc(N*sizeof(float));
  p3_prime = (float*)malloc(N*sizeof(float));
  diverged = (int*)malloc(N*sizeof(int));

  p_dv_1_x = (float*)malloc(N*sizeof(int));
  p_dv_1_y = (float*)malloc(N*sizeof(int));
  p_dv_1_z = (float*)malloc(N*sizeof(int));

  p_dv_2_x = (float*)malloc(N*sizeof(int));
  p_dv_2_y = (float*)malloc(N*sizeof(int));
  p_dv_2_z = (float*)malloc(N*sizeof(int));

  p_dv_3_x = (float*)malloc(N*sizeof(int));
  p_dv_3_y = (float*)malloc(N*sizeof(int));
  p_dv_3_z = (float*)malloc(N*sizeof(int));

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
  cudaMalloc(&d_diverged, N*sizeof(int));

  cudaMalloc(&d_p_dv_1_x, N*sizeof(float));
  cudaMalloc(&d_p_dv_1_x, N*sizeof(float));
  cudaMalloc(&d_p_dv_1_x, N*sizeof(float));

  cudaMalloc(&d_p_dv_2_x, N*sizeof(float));
  cudaMalloc(&d_p_dv_2_y, N*sizeof(float));
  cudaMalloc(&d_p_dv_2_z, N*sizeof(float));

  cudaMalloc(&d_p_dv_3_x, N*sizeof(float));
  cudaMalloc(&d_p_dv_3_y, N*sizeof(float));
  cudaMalloc(&d_p_dv_3_z, N*sizeof(float));

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













