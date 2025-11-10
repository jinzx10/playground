#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <ratio>
#include <cassert>
#include "cuda_runtime.h"
#include "thrust/device_vector.h"

using iclock = std::chrono::steady_clock;

__device__ __host__ float isin(float x, int nterm) {
    // sin(x) = x - x^3/3! + x^5/5! - ...
    float term = x;
    float sum = x;
    float x2 = x * x;
    for (int n = 1; n < nterm; n++) {
        term *= -x2 /(2*n*(2*n+1));
        sum += term;
    }
    return sum;
}

__global__ void isin_many(float* val, float width, int ngrid, int nterm) {
    int igrid = blockIdx.x * blockDim.x + threadIdx.x;
    if (igrid < ngrid) {
        val[igrid] = isin(igrid * width, nterm);
    }
}


int main(int argc, char** argv) {

    assert(argc >= 3);
    int ngrid = atoi(argv[1]);
    int nterm = atoi(argv[2]);
    int threads = 32;
    int blocks = (ngrid + threads - 1) / threads; // round up

    double pi = 3.14159265358979323846;
    double width = pi / (ngrid-1); // width of each trapezoid

    thrust::device_vector<float> dev_val(ngrid);
    float* val = thrust::raw_pointer_cast(&dev_val[0]);
    iclock::time_point start = iclock::now();
    isin_many<<<blocks, threads>>>(val, width, ngrid, nterm); 
    double sum = thrust::reduce(dev_val.begin(), dev_val.end());
    std::chrono::duration<double, std::milli> dur = iclock::now() - start;

    sum -= 0.5 * (isin(0, nterm) + isin(pi, nterm)); // trapezoid correction
    sum *= width;

    printf("ngrid=%d, nterm=%d\n", ngrid, nterm);
    printf("result : %20.10f\n", sum);
    printf("elapsed: %8.1f\n", dur.count());
}
