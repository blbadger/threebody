#include <stdio.h>
#include <iostream>

__global__
void divergence(int n, 
              float *p1, 
              float *p2, 
              float *p3, 
              float *p1_prime, 
              float *p2_prime, 
              float *p3_prime, 
              int *diverged,
              float *p_dv_1,
              float *p_dv_2, 
              float *p_dv_3)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  for (int j=0; j < 50000; j++) {
    if (i < n) {
      p_dv_1[i] = -9.8f * 10 * (p1[i] - p2[i])/((p1[i] - p2[i])*(p1[i] - p2[i]));
      p_dv_2[i] = -9.8f * 10 * (p2[i] - p3[i])/((p2[i] - p3[i])*(p2[i] - p3[i]));
      p_dv_3[i] = -9.8f * 10 * (p1[i] - p3[i])/((p1[i] - p3[i])*(p1[i] - p3[i]));
      p1[i] = p1[i] + 0.01f * p_dv_1[i];
      p2[i] = p2[i] + 0.01f * p_dv_2[i];
      p3[i] = p3[i] + 0.01f * p_dv_3[i];
      if (p1[i] - p1_prime[i] > 1 || p2[i] - p2_prime[i] > 1 || p3[i] - p3_prime[i] > 1){
        diverged[i] = i;
      } 
    }

  }
}

int main(void)
{
  int N = 1<<18;
  std::cout << N;
  float *p1, *p2, *p3, *p1_prime, *p2_prime, *p3_prime, *d_p1, *d_p2, *d_p3, *d_p1_prime, *d_p2_prime, *d_p3_prime;
  int *diverged, *d_diverged;
  float *p_dv_1, *p_dv_2, *p_dv_3, *d_p_dv_1, *d_p_dv_2, *d_p_dv_3;
  p1 = (float*)malloc(N*sizeof(float));
  p2 = (float*)malloc(N*sizeof(float));
  p3 = (float*)malloc(N*sizeof(float));

  p1_prime = (float*)malloc(N*sizeof(float));
  p2_prime = (float*)malloc(N*sizeof(float));
  p3_prime = (float*)malloc(N*sizeof(float));
  diverged = (int*)malloc(N*sizeof(int));
  p_dv_1 = (float*)malloc(N*sizeof(int));
  p_dv_2 = (float*)malloc(N*sizeof(int));
  p_dv_3 = (float*)malloc(N*sizeof(int));

  cudaMalloc(&d_p1, N*sizeof(float)); 
  cudaMalloc(&d_p2, N*sizeof(float));
  cudaMalloc(&d_p3, N*sizeof(float));
  cudaMalloc(&d_p1_prime, N*sizeof(float));
  cudaMalloc(&d_p2_prime, N*sizeof(float));
  cudaMalloc(&d_p3_prime, N*sizeof(float));
  cudaMalloc(&d_diverged, N*sizeof(int));

  cudaMalloc(&d_p_dv_1, N*sizeof(float));
  cudaMalloc(&d_p_dv_2, N*sizeof(float));
  cudaMalloc(&d_p_dv_3, N*sizeof(float));

  for (int i = 0; i < N; i++) {
    p1[i] = 0.0f + i;
    p2[i] = 0.0f + i;
    p3[i] = 0.0f + i;
    p1_prime[i] = 0.0f + i + 0.0001f;
    p2_prime[i] = 0.0f + i;
    p3_prime[i] = 0.0f + i;
    diverged[i] = 0;
    p_dv_1[i] = 0;
    p_dv_2[i] = 0;
    p_dv_3[i] = 0;

  }

  cudaMemcpy(d_p1, p1, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_p2, p2, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_p3, p3, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_p1_prime, p1_prime, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_p2_prime, p2_prime, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_p3_prime, p3_prime, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_diverged, diverged, N*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_p_dv_1, p_dv_1, N*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_p_dv_2, p_dv_2, N*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_p_dv_3, p_dv_3, N*sizeof(int), cudaMemcpyHostToDevice);


  // call CUDA kernal
  divergence<<<(N+255)/256, 256>>>(N, d_p1, d_p2, d_p3, d_p1_prime, d_p2_prime, d_p3_prime, d_diverged, d_p_dv_1, d_p_dv_2, d_p_dv_3);

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

  free(p1);
  free(p2);
  free(p3);
  free(p1_prime);
  free(p2_prime);
  free(p3_prime);
}