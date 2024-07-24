#include <gsl/gsl_interp.h>
#include <cmath>
#include <iostream>

int main() {

    double rmax = 10;
    int nr = 1000;
    double dr = rmax / (nr - 1);

    double* r = new double[nr];
    double* f = new double[nr];

    for (int ir = 0; ir != nr; ++ir) {
        r[ir] = ir * dr;
        f[ir] = std::cos(r[ir]);
    }

    int nfit = 4;

    // evaluate the interpolant at R
    double R = 3.00;

    int istart = int(R / dr) - nfit / 2;

    if (istart < 0)
        istart = 0;
    
    if (istart > nr-nfit)
        istart = nr-nfit;

    gsl_interp_accel* acc = gsl_interp_accel_alloc();
    gsl_interp* interp = gsl_interp_alloc(gsl_interp_polynomial, nfit);
    gsl_interp_init(interp, &r[istart], &f[istart], nfit);


    printf("%20.15e\n", gsl_interp_eval(interp, &r[istart], &f[istart], R, acc));




    return 0;
}
