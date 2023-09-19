#include <stdio.h>
#include <iostream>
#include <time.h>
#include <sys/time.h>
#define USECPSEC 1000000ULL

unsigned long long myCPUTimer(unsigned long long start=0){
  timeval tv;
  gettimeofday(&tv, 0);
  return ((tv.tv_sec*USECPSEC)+tv.tv_usec)-start;
}

// kernal code
__global__
void saxpy(int n, float a, float *x, float *y)
{
  
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  for (int j=0; j < 50000; j++){
    if (i < n) y[i] = a*x[i] + y[i];
}

}

int main(void)
{
  int N = 1<<24;
  std::cout << N << '\n';
  float *x, *y, *d_x, *d_y;
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));

  cudaMalloc(&d_x, N*sizeof(float)); 
  cudaMalloc(&d_y, N*sizeof(float));

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

  // Perform SAXPY on 1M elements
  long long t1 = myCPUTimer();
  saxpy<<<(N+255)/256, 256>>>(N, 2.0, d_x, d_y);
  // cudaDeviceSynchronize();
  long long t2 = myCPUTimer();
  std::cout << t2 - t1 << '\n';

  cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

  float maxError = 0.0f;
  for (int i = 0; i < 100; i++)
    std::cout << y[i] << ' ';
  printf("Max error: %f\n", maxError);

  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);
}
