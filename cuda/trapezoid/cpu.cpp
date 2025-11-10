#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <ratio>
#include <cassert>
#include <omp.h>

using iclock = std::chrono::steady_clock;

inline float isin(float x, int nterm) {
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


int main(int argc, char** argv) {

    assert(argc >= 3);
    int ngrid = atoi(argv[1]);
    int nterm = atoi(argv[2]);
    int num_threads = (argc >= 4) ? atoi(argv[3]) : 1;

    double pi = 3.14159265358979323846;
    double sum = 0.0;
    double width = pi / (ngrid-1); // width of each trapezoid

    iclock::time_point start = iclock::now();
    omp_set_num_threads(num_threads);
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < ngrid; i++) {
        sum += isin(i * width, nterm);
    }
    std::chrono::duration<double, std::milli> dur = iclock::now() - start;

    sum -= 0.5 * (isin(0, nterm) + isin(pi, nterm)); // trapezoid correction
    sum *= width;

    printf("ngrid=%d, nterm=%d\n", ngrid, nterm);
    printf("result : %20.10f\n", sum);
    printf("elapsed: %8.1f\n", dur.count());
}
