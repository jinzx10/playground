#include <cstdio>
#include <cmath>
#include "../clock.h"


void plain(double* a, double* b, double* c, int nelem) {
    for (int i = 0; i < nelem; i++) {
        c[i] = a[i] + b[i];
    }
}

void omp(double* a, double* b, double* c, int nelem) {
    #pragma omp parallel for
    for (int i = 0; i < nelem; i++) {
        c[i] = a[i] + b[i];
    }
}


int main() {

    int nelem = 1e7;
    size_t nbytes = sizeof(double) * nelem;

    double* buf = new double[nbytes * 3];

    double* ha = buf;
    double* hb = ha + nelem;
    double* res = hb + nelem;

    for (int i = 0; i < nelem; i++) {
        ha[i] = std::sin(i) * std::sin(i);
        hb[i] = std::cos(i) * std::cos(i);
    }

    CLOCK(plain(ha, hb, res, nelem), "plain");
    CLOCK(omp(ha, hb, res, nelem), "omp");

    for (int i = 0; i < nelem; i++) {
        if (std::abs(res[i] - 1) > 1e-12) {
            printf("mismatch at %d: %f\n", i, res[i]);
            break;
        }
    }

    delete[] buf;
}


