#include <cstdio>

__global__ void print_thread_index() {
    printf("gridDim: (%d,%d,%d), blockDim: (%d,%d,%d), blockIdx: (%d,%d,%d), threadIdx: (%d,%d,%d)\n",
           gridDim.x, gridDim.y, gridDim.z,
           blockDim.x, blockDim.y, blockDim.z,
           blockIdx.x, blockIdx.y, blockIdx.z,
           threadIdx.x, threadIdx.y, threadIdx.z);
}


int main() {

    dim3 grid(3, 2, 1); // number of blocks in a grid
    dim3 block(1, 2, 3); // number of threads in a block
    print_thread_index<<<grid, block>>>();

    cudaDeviceSynchronize();

    return 0;
}
