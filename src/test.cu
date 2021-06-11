#include "test.cuh"

__global__
void add(int n, double *x, double *y, double *z)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    z[i] = x[i] + y[i];
}

namespace CudaWrappers{

    void  add_wrapper(int N, double *x, double *y, double *z, int blockSize) {
    
        // Allocate on GPU
        double *dx, *dy, *dz;
        int size = N * sizeof(double);
    
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        cudaMalloc((void **) &dx, size);
        cudaMalloc((void **) &dy, size);
        cudaMalloc((void **) &dz, size);
        cudaMemcpy(dx, x, size, cudaMemcpyHostToDevice);
        cudaMemcpy(dy, y, size, cudaMemcpyHostToDevice);
        cudaMemcpy(dz, z, size, cudaMemcpyHostToDevice);

    
        // Run kernel on 1M elements on the GPU
        int numBlocks = (N + blockSize - 1) / blockSize;
        add<<<numBlocks, blockSize>>>(N, dx, dy, dz);

        cudaError_t mycudaerror = cudaGetLastError() ;
    
        cudaMemcpy(z, dz, N*sizeof(double), cudaMemcpyDeviceToHost);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
       
        cudaFree(dx);
        cudaFree(dy);
        cudaFree(dz);

    }

}
