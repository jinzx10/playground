#include <cstdio>
#include <cmath>
#include "../clock.h"
#include "../check.h"

#include <chrono>
using iclock = std::chrono::high_resolution_clock;

void plain(double* a, double* b, double* c, int nelem) {
    for (int i = 0; i < nelem; i++) {
        c[i] = a[i] + b[i];
    }
}

// nvcc --compiler-options -fopenmp
//void omp(double* a, double* b, double* c, int nelem) {
//    #pragma omp parallel for
//    for (int i = 0; i < nelem; i++) {
//        c[i] = a[i] + b[i];
//    }
//}


__global__ void gpu(double* a, double* b, double* c, int nx, int ny) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = ix + iy * nx;

    if (ix < nx && iy < ny) {
        c[idx] = a[idx] + b[idx];
    }
}


int main() {

    iclock::time_point start;
    std::chrono::duration<double> dur;

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev), "device prop");
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev), "set dev");

    int nx = 1 << 13;
    int ny = 1 << 13;
    int nelem = nx * ny;
    size_t nbytes = sizeof(double) * nelem;

    double* buf = new double[nbytes * 4];

    double* ha = buf;
    double* hb = ha + nelem;
    double* cpu_res = hb + nelem;
    double* gpu_res = cpu_res + nelem;

    for (int i = 0; i < nelem; i++) {
        ha[i] = std::sin(i) * std::sin(i);
        hb[i] = std::cos(i) * std::cos(i);
    }

    CLOCK(plain(ha, hb, cpu_res, nelem), "plain");
    //CLOCK(omp(ha, hb, cpu_res, nelem), "omp");

    double *da, *db, *dc;
    cudaMalloc((double**)&da, nbytes);
    cudaMalloc((double**)&db, nbytes);
    cudaMalloc((double**)&dc, nbytes);

    cudaMemcpy(da, ha, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, nbytes, cudaMemcpyHostToDevice);

    dim3 grid(256, 256);
    dim3 block(32, 32);

    start = iclock::now();
    gpu<<<grid, block>>>(da, db, dc, nx, ny);
    cudaDeviceSynchronize();
    dur = iclock::now() - start;
    printf("gpu elapsed time: %f\n", dur.count());

    cudaMemcpy(gpu_res, dc, nbytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < nelem; i++) {
        if (std::abs(cpu_res[i] - gpu_res[i]) > 1e-12) {
            printf("mismatch at %d: %f %f\n", i, cpu_res[i], gpu_res[i]);
            break;
        }
    }


    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    delete[] buf;
}


