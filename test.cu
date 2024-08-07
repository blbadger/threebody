
#include <stdio.h>
#include <iostream>
#include <chrono>
#include <fstream>
#include <iterator>
#include <fstream>
#include <omp.h>

#include <cuda.h>
#include <cuda_runtime_api.h>


__global__
void saxpy(int n, float a, float *x)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  for (int j=0; j < 10000000; j++){
    if (i < n) {
      x[i] += 1;
    }
  }
}

int main(void) {
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  float *x, *d_x;

  int N = 10000000;
  cudaHostAlloc((void**)&x, N*sizeof(float), cudaHostAllocWriteCombined | cudaHostAllocMapped);

  int n_gpus=4;
  for (int i = 0; i < N/n_gpus; i++) {
      x[i] = 1.0f;
  }

  cudaStream_t streams[n_gpus];
  #pragma omp parallel num_threads(n_gpus)
  {
    int d=omp_get_thread_num();
    std::cout << "Thread" << d << "activated\n";
    std::cout << d << " Device initialized \n";
    int start_idx = (N/n_gpus)*omp_get_thread_num();
    int end_idx = start_idx + (N/n_gpus);
    std::cout << "Start index: " << start_idx << "\n";
    std::cout << "End index: " << end_idx << "\n";
    cudaSetDevice(omp_get_thread_num());
    cudaStreamCreate(&streams[d]);

    cudaMalloc(&d_x, (N/n_gpus)*sizeof(float));
    cudaMemcpyAsync(d_x, x+start_idx, (N/n_gpus)*sizeof(float), cudaMemcpyHostToDevice, streams[d]);

    saxpy<<<(N+127)/128, 128, 0, streams[d]>>>(N/n_gpus, 2.0f, d_x);

    cudaMemcpyAsync(x+start_idx, d_x, (N/n_gpus)*sizeof(float), cudaMemcpyDeviceToHost, streams[d]);
    cudaDeviceSynchronize();
  }

  cudaError_t err = cudaGetLastError();  // add
  if (err != cudaSuccess) std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl; // add
  std::cout << x[0] << "\n";

  // measure elapsed kernal runtime
  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  std::cout << "Elapsed Time: " << elapsed_seconds.count() << "s\n";

  return;
}