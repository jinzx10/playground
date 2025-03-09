#include <cstdio>
#include <cmath>
#include "../clock.h"
#include "../check.h"


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


__global__ void gpu(double* a, double* b, double* c, int nelem) {
    // assuming grid size is 1 and block is one-dimension
    for (int i = threadIdx.x; i < nelem; i += blockDim.x)
        c[i] = a[i] + b[i];
}


int main() {

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev), "device prop");
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev), "set dev");

    int nelem = 1e7;
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

    dim3 block(128);
    gpu<<<1, block>>>(da, db, dc, nelem);

    cudaMemcpy(gpu_res, dc, nbytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < nelem; i++) {
        if (std::abs(cpu_res[i] - gpu_res[i]) > 1e-12) {
            printf("mismatch at %d: %f %f\n", i, cpu_res[i], gpu_res[i]);
            break;
        }
    }





    //cudaFree(dbuf);
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    delete[] buf;
}


