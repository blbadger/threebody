#include <stdio.h>
#include <iostream>
#include <chrono>
#include <fstream>
#include <iterator>
#include <fstream>

#include <cuda.h>
#include <cuda_runtime_api.h>


__global__
void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  for (int j=0; j < 10000000; j++){
    if (i < n) {
      y[i] = 20 + y[i];
    }
  }
}

// #define SHIFT_AMOUNT 16 // 2^16 = 65536
// #define SHIFT_MASK ((1 << SHIFT_AMOUNT) - 1) // 65535 (all LSB set, all MSB clear)

int main(void) {
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  float *x, *y, *d_x, *d_y;
  int N = 1<<20;
  for (int d=0; d<2; d++){
    std::cout << d << " Device initialized \n";
    cudaSetDevice(d);
    cudaHostAlloc((void**)&x, N*sizeof(float), cudaHostAllocWriteCombined | cudaHostAllocMapped );
    cudaHostAlloc((void**)&y, N*sizeof(float), cudaHostAllocWriteCombined | cudaHostAllocMapped );

    cudaMalloc(&d_x, N*sizeof(float)); 
    cudaMalloc(&d_y, N*sizeof(float));

    for (int i = 0; i < N; i++) {
      x[i] = 1.0f;
      y[i] = 2.0f;
    }

    cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

    // Perform SAXPY on 1M elements
    saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);
    cudaMemcpyAsync(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

    // cudaFree(d_x);
    // cudaFree(d_y);
    // free(x);
  }

  cudaDeviceSynchronize();
  std::cout << y[0];

  // measure elapsed kernal runtime
  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  std::cout << "Elapsed Time: " << elapsed_seconds.count() << "s\n";

  
  return;
}

extern "C" {
float* save_arr()
  {
    int N = 1<<20;
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

    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();

    // Perform SAXPY on 1M elements
    saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);

    cudaDeviceSynchronize();

    // measure elapsed kernal runtime
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Elapsed Time: " << elapsed_seconds.count() << "s\n";

    cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
    free(x);
    return y;
  }
}