#include <cuda.h>
#include <cuda_runtime_api.h>

__global__ void cuda_sum_kernel(float *a, float *b, float *c, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) {
        return;
    }

    c[idx] = a[idx] + b[idx];
}

extern "C" {
void cuda_sum(float *a, float *b, float *c, size_t size)
{
    float *d_a, *d_b, *d_c;

    cudaMalloc((void **)&d_a, size * sizeof(float));
    cudaMalloc((void **)&d_b, size * sizeof(float));
    cudaMalloc((void **)&d_c, size * sizeof(float));

    cudaMemcpy(d_a, a, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size * sizeof(float), cudaMemcpyHostToDevice);

    cuda_sum_kernel <<< ceil(size / 256.0), 256 >>> (d_a, d_b, d_c, size);

    cudaMemcpy(c, d_c, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
}